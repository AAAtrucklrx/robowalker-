#!/usr/bin/env python3
"""标定板检测：从图像里找出特征点的亚像素位置。

任务书允许棋盘格 / ArUco / AprilTag / 任何已知几何的目标。本模块实现两类：

* **棋盘格 (chessboard)** —— 默认。内角点检测 + 亚像素细化。
* **ChArUco** —— 方格 + ArUco。**部分出画或局部遮挡时仍能检测**，
  对采集动作的容错高得多，强烈建议有条件时改用它。

一个关键区别（也是新手最容易搞错的）
------------------------------------------------------------------
``pattern_size`` 是**内角点数**，不是方格数：

    棋盘格 10x7 个方格  ->  pattern_size = (9, 6)

``square_size`` 必须与**打印实物**一致，单位米。它写错不会让内参错，
但会让平移外参的尺度整体错（任务书 5.3 专门点了这一条）。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BoardSpec:
    """标定板的几何定义。所有参数都来自配置文件，不硬编码。"""

    type: str = "chessboard"            # chessboard | charuco
    pattern_size: tuple[int, int] = (9, 6)   # 内角点 (列, 行)
    square_size: float = 0.025          # 米
    marker_size: float = 0.0            # charuco: marker 边长（米）
    dict_name: str = "DICT_5X5_100"     # charuco 字典

    @classmethod
    def from_config(cls, cfg: dict) -> "BoardSpec":
        t = cfg.get("target", {})
        typ = t.get("type", "chessboard")
        if typ == "chessboard":
            c = t.get("chessboard", {})
            return cls("chessboard", tuple(c.get("pattern_size", (9, 6))),
                       float(c.get("square_size", 0.025)))
        if typ in ("aruco", "charuco"):
            a = t.get("aruco", {})
            ps = tuple(a.get("pattern_size", (5, 7)))
            sq = float(a.get("square_size", a.get("marker_size", 0.05)))
            return cls("charuco", ps, sq,
                       float(a.get("marker_size", sq * 0.75)),
                       a.get("dict", "DICT_5X5_100"))
        raise ValueError(f"暂不支持的标定板类型: {typ}")

    # -- 几何 --------------------------------------------------
    def object_points(self) -> np.ndarray:
        """板坐标系下的 3D 特征点，z=0 平面，原点在左上角第一个内角点。

        约定：x 向右、y 向下、z 垂直板面向外。返回值 shape=(N,3) float32。
        """
        cols, rows = self.pattern_size
        s = self.square_size
        # 注意 y 向下，与图像坐标系一致（OpenCV 的 findChessboardCorners
        # 也是从左上角开始按行返回角点）
        pts = np.zeros((cols * rows, 3), np.float32)
        pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * s
        return pts

    def expected_corners(self) -> int:
        return self.pattern_size[0] * self.pattern_size[1]

    def __str__(self):
        c, r = self.pattern_size
        return (f"{self.type} 内角点 {c}x{r}，方格 {self.square_size*1000:.2f} mm"
                f"（板面 {(c+1)*self.square_size*1000:.0f}x"
                f"{(r+1)*self.square_size*1000:.0f} mm）")


@dataclass
class Detection:
    """一张图的检测结果。"""

    found: bool
    image_points: np.ndarray | None = None      # (N,1,2) float32
    object_points: np.ndarray | None = None     # (N,3) float32
    method: str = ""
    marker_ids: np.ndarray | None = None        # charuco 才有


# ══════════════════════════════════════════════════════════════
#  棋盘格
# ══════════════════════════════════════════════════════════════
def _sb_flags():
    """findChessboardCornersSB 只接受 EXHAUSTIVE/ACCURACY/LARGER/MARKER，
    传经典检测器的 flag（ADAPTIVE_THRESH 等）会直接抛异常。"""
    f = 0
    for name in ("CALIB_CB_EXHAUSTIVE", "CALIB_CB_ACCURACY"):
        f |= getattr(cv2, name, 0)
    return f


def _classic_flags():
    f = 0
    for name in ("CALIB_CB_ADAPTIVE_THRESH", "CALIB_CB_NORMALIZE_IMAGE"):
        f |= getattr(cv2, name, 0)
    return f


def _reorder_to_match_detector(op: np.ndarray, spec: BoardSpec, detector: str):
    """把 object points 重排成**与检测器返回顺序一致**。

    ⚠️ 这是本项目最隐蔽的一个 bug，必须讲清楚：

    ``findChessboardCornersSB`` 返回的角点顺序在 **x 方向与
    ``object_points()`` 的约定相反**。如果直接把两者配对，得到的是一个
    **镜像对应**——而棋盘格在 x 反序下是**对称**的，所以
    ``calibrateCamera`` **照样能收敛、残差看起来也不大**，但内参会系统性偏掉。

    实测（合成数据，真值 fx=1100）：
      · 不重排：fx = 1135.66（**偏 +3.2%**），与 ROS camera_calibration
        （fx = 1099.53，几乎命中真值）在投影意义上差 RMS 18.4 px
      · 重排后：与 ROS 一致

    **是"与成熟工具对比"这一项验证把这个 bug 抓出来的** —— 只跑自检
    发现不了，因为自检的渲染器和检测器用的是同一个错误约定。

    经典 ``findChessboardCorners`` 用的是常规行主序（与 ``object_points()``
    一致），所以只对 SB 做重排。
    """
    if detector != "SB":
        return op
    cols, rows = int(spec.pattern_size[0]), int(spec.pattern_size[1])
    if op.shape[0] != cols * rows:
        return op
    return np.ascontiguousarray(op.reshape(rows, cols, 3)[:, ::-1].reshape(-1, 3))


def detect_chessboard(gray: np.ndarray, spec: BoardSpec,
                      refine: bool = True, allow_classic: bool = True) -> Detection:
    """检测棋盘格内角点，返回**角点与 3D 点已正确配对**的检测结果。

    优先用 ``findChessboardCornersSB``（OpenCV 4.x 的 sector-based 检测器，
    自带亚像素精度、对噪声和模糊鲁棒得多）；不可用时退回
    ``findChessboardCorners`` + ``cornerSubPix``。

    返回的 ``object_points`` 已经按检测器的顺序重排过，调用方（内参标定、
    PnP）直接配对即可，**不要再自己反序**。

    ⚠️ ``allow_classic``：一个**必须在实时场景关掉**的性能陷阱
    ------------------------------------------------------------------
    实测（1624x1240）：

    ============================================  =========
    ``findChessboardCornersSB``                   **71 ms**
    ``findChessboardCorners``（经典）              **684 ms**
    ``detect()`` 完整（SB 失败 → 回退经典）        **919 ms**
    ============================================  =========

    也就是说**板子没找到的时候反而慢 13 倍**（71 → 919 ms），因为那时才回退到
    经典检测器。而"板子没找到"正是**实时取景一边瞄准一边看**的状态 ——
    最需要流畅的时候偏偏卡成 ~1 fps，**看起来就像"相机帧率太低"**。

    所以：**离线标定**（``run_calibration``）保留回退 —— 多找回几帧是有价值
    的，900 ms 一帧无所谓；**实时取景 / 采集反馈**必须传
    ``allow_classic=False`` 只走 SB 快路径。少检测一帧对实时提示无害
    （下一帧还会再测），但卡顿会直接毁掉瞄准与对焦体验。
    """
    ps = tuple(int(v) for v in spec.pattern_size)
    op = spec.object_points()

    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, ps, flags=_sb_flags())
        if ok:
            c = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
            # SB 检测器对某些图像会多返回一组首尾重复点，剔掉
            if len(c) > len(op):
                c = c[: len(op)]
            return Detection(True, c,
                             _reorder_to_match_detector(op, spec, "SB"),
                             method="SB")
        if not allow_classic:
            return Detection(False, None, None, method="SB")

    ok, corners = cv2.findChessboardCorners(gray, ps, flags=_classic_flags())
    if not ok:
        return Detection(False, None, None, method="classic")
    if refine:
        # 亚像素细化：粗检测的角点误差能差一个量级，这一步不能省
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return Detection(True, corners.astype(np.float32),
                     _reorder_to_match_detector(op, spec, "classic"),
                     method="classic+subpix")


# ══════════════════════════════════════════════════════════════
#  ChArUco（兼容 OpenCV 4.6 与 4.7+ 两套 API）
# ══════════════════════════════════════════════════════════════
def _aruco_dict(name: str):
    a = cv2.aruco
    if not hasattr(a, name):
        raise ValueError(f"OpenCV 没有字典 {name}")
    getter = getattr(a, "getPredefinedDictionary", None)
    return getter(getattr(a, name)) if getter else a.Dictionary_get(getattr(a, name))


def make_charuco_board(spec: BoardSpec):
    a, cv = cv2.aruco, cv2
    d = _aruco_dict(spec.dict_name)
    ms = spec.marker_size or spec.square_size * 0.75
    if hasattr(a, "CharucoBoard"):
        return a.CharucoBoard(spec.pattern_size, spec.square_size, ms, d)
    return a.CharucoBoard_create(spec.pattern_size[0], spec.pattern_size[1],
                                 spec.square_size, ms, d)


def detect_charuco(gray: np.ndarray, spec: BoardSpec) -> Detection:
    a = cv2.aruco
    board = make_charuco_board(spec)
    d = _aruco_dict(spec.dict_name)
    params = a.DetectorParameters() if hasattr(a, "DetectorParameters") \
        else a.DetectorParameters_create()

    if hasattr(a, "CharucoDetector"):
        det = a.CharucoDetector(board)
        ch_corners, ch_ids, mk_corners, mk_ids = det.detectBoard(gray)
    else:
        mk_corners, mk_ids, _ = a.detectMarkers(gray, d, parameters=params)
        if mk_ids is None or len(mk_ids) < 4:
            return Detection(False, method="charuco")
        _, ch_corners, ch_ids = a.interpolateCornersCharuco(
            mk_corners, mk_ids, gray, board)

    if ch_ids is None or len(ch_ids) < 6:
        return Detection(False, method="charuco")

    ch_ids = np.asarray(ch_ids).reshape(-1)
    ch_corners = np.asarray(ch_corners, dtype=np.float32).reshape(-1, 1, 2)
    # ChArUco 只检测到部分角点是常态：只取检到的那些 3D 点
    all_obj = board.getChessboardCorners().astype(np.float32) \
        if hasattr(board, "getChessboardCorners") else board.chessboardCorners
    obj = all_obj[ch_ids]
    return Detection(True, ch_corners, obj, method="charuco",
                     marker_ids=np.asarray(mk_ids).reshape(-1))


# ══════════════════════════════════════════════════════════════
#  统一入口
# ══════════════════════════════════════════════════════════════
def detect(gray: np.ndarray, spec: BoardSpec,
           allow_classic: bool = True) -> Detection:
    """检测标定板。``allow_classic=False`` 走快路径（实时场景用，见
    :func:`detect_chessboard` 的性能说明）。"""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if spec.type == "chessboard":
        return detect_chessboard(gray, spec, allow_classic=allow_classic)
    if spec.type == "charuco":
        return detect_charuco(gray, spec)
    raise ValueError(f"未知标定板类型 {spec.type}")


def solve_pnp_board(object_points, image_points, K, dist, prev_R=None,
                    prev_t=None, pattern_size=None,
                    corner_order: str = "canonical") -> dict:
    """解标定板位姿：**顺序自适应** + **平面二义性消解**。

    ⚠️ 两个都必须做，而且它们是**两个不同的问题**，别混为一谈
    ------------------------------------------------------------------
    **(A) 角点顺序约定不一致**

    ``findChessboardCornersSB`` 返回的角点顺序与本模块 ``object_points()``
    的约定**在 x 方向相反**。实测（合成数据、120 帧）：按原顺序喂 PnP，
    重投影 RMS 是 **108.8 px**；把 x 反序后降到 **0.13 px**，而且 120/120
    帧都是同一个结论 —— 是**固定约定差异**，不是随机噪声。

    应对：**必须固定用哪一种顺序**，不能"两种都试然后按重投影误差选"——
    实测那样做会全崩，因为棋盘格在 x 反序下是对称的，**两种顺序都能找到
    重投影一样好的位姿**（镜像二义性），按误差选等于抛硬币。
    **但请注意**：``detect()`` 现在已经在源头把 object points 重排成与检测器
    一致了（见 ``_reorder_to_match_detector``），所以这里**默认不再重复反序**。
    参数 ``corner_order="x_reversed"`` 只留给"直接喂原始 object_points"的场景。

    **(B) 平面目标二义性**

    平面标定板存在两解。直接取 ``cv2.solvePnP`` 的返回值有相当概率拿到
    翻转的那个。对 C1 内参无害，但对 C2/C3 的影响要看情况：

    * **旋转外参不受影响** —— AX=XB 里 ``B = R_CB(2)·S·Sᵀ·R_CB(1)ᵀ``，
      常量板系偏置 S 会抵消（实测：位姿整体差 179.8° 时 R_IC 仍只差 0.22°）；
    * **平移/杠杆臂会崩** —— 因为 ``a_WC``、``R̈`` 都在板系里，而重力方向
      在板系中是未知的。所以 ``solve_lever_arm`` 里把重力也当未知量联合估计。

    判据：板心在相机前方（``t[2] > 0``）且板面朝向相机（``R[2,2] < 0``）；
    候选重投影接近时优先取与上一帧时间连续的。

    返回 dict：``rvec``、``tvec``、``order``（用了哪种顺序）、
    ``n_candidates``、``picked_err``。
    """
    objp = np.asarray(object_points, np.float32)

    if corner_order == "x_reversed" and pattern_size is not None:
        cols, rows = int(pattern_size[0]), int(pattern_size[1])
        if objp.shape[0] == cols * rows:
            objp = np.ascontiguousarray(objp.reshape(rows, cols, 3)[:, ::-1]
                                        .reshape(-1, 3))
            return {**_solve_pnp_one_order(objp, image_points, K, dist,
                                           prev_R, prev_t),
                    "order": "x_reversed"}
    r = _solve_pnp_one_order(objp, image_points, K, dist, prev_R, prev_t)
    r["order"] = "canonical"
    return r


def _solve_pnp_one_order(object_points, image_points, K, dist, prev_R=None,
                         prev_t=None) -> dict:
    """在**给定角点顺序**下解位姿并消解平面二义性（被上面那个函数调用）。"""
    objp = np.asarray(object_points, np.float32)
    imgp = np.asarray(image_points, np.float32)

    cands = []
    getter = getattr(cv2, "solvePnPGeneric", None)
    if getter is not None:
        try:
            ok, rvecs, tvecs, errs = getter(
                objp, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE)
            if ok and len(rvecs):
                for rv, tv, e in zip(rvecs, tvecs, errs):
                    R, _ = cv2.Rodrigues(np.asarray(rv, float))
                    cands.append((R, np.asarray(tv, float).reshape(3),
                                  float(np.asarray(e).ravel()[0])))
        except cv2.error:
            cands = []

    if not cands:
        ok, rv, tv = cv2.solvePnP(objp, imgp, K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError("solvePnP 失败")
        R, _ = cv2.Rodrigues(rv)
        cands = [(R, tv.reshape(3), float("nan"))]

    # ── 关键一步：每个候选都要 LM 细化 ─────────────────────────
    # IPPE 是闭式解，没有非线性细化，精度明显低于迭代法（实测残差能从
    # 0.6° 涨到 8.9°）。但迭代法又需要好的初值且自己分不出两个分支。
    # 所以正确组合是：**IPPE 给出两个分支 → 各自 LM 细化 → 再按判据选**。
    refined = []
    for R, t, e in cands:
        rv, _ = cv2.Rodrigues(R)
        tv = t.reshape(3, 1).copy()
        rvc = rv.copy()
        try:
            if hasattr(cv2, "solvePnPRefineLM"):
                rvc, tv = cv2.solvePnPRefineLM(objp, imgp, K, dist, rvc, tv)
            else:
                _, rvc, tv = cv2.solvePnP(objp, imgp, K, dist, rvc, tv, True,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            pass
        R2, _ = cv2.Rodrigues(rvc)
        proj, _ = cv2.projectPoints(objp, rvc, tv, K, dist)
        err = float(np.sqrt(np.mean(np.sum(
            (proj.reshape(-1, 2) - imgp.reshape(-1, 2)) ** 2, axis=1))))
        refined.append((R2, tv.reshape(3), err))
    cands = refined

    def ok_front(c):
        return c[1][2] > 0 and c[0][2, 2] < 0

    front = [c for c in cands if ok_front(c)]
    pool = front or [c for c in cands if c[1][2] > 0] or cands

    if prev_R is not None and len(pool) > 1:
        prev_R = np.asarray(prev_R, float)
        # 重投影误差接近（1.5 倍内）时，取与上一帧最接近的解
        best_e = min((c[2] for c in pool if np.isfinite(c[2])), default=0.0)
        near = [c for c in pool if not np.isfinite(c[2]) or c[2] <= best_e * 1.5 + 1e-9]
        pool = near or pool
        pool = sorted(pool, key=lambda c: rot_angle(c[0].T @ prev_R))
    else:
        pool = sorted(pool, key=lambda c: (c[0][2, 2], c[2]))

    R, t, e = pool[0]
    rvec, _ = cv2.Rodrigues(R)
    return {
        "rvec": rvec.reshape(3, 1),
        "tvec": t.reshape(3, 1),
        "n_candidates": len(cands),
        "n_front_facing": len(front),
        "picked_err": e,
        "all_err": [c[2] for c in cands],
        "ambiguous": len(cands) > 1,
    }


def rot_angle(R: np.ndarray) -> float:
    """旋转矩阵的旋转角（弧度）。"""
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def draw_detection(gray: np.ndarray, det: Detection, spec: BoardSpec) -> np.ndarray:
    """画检测结果，用于肉眼确认（写报告插图也用它）。

    ⚠️ **必须按下标点个数保护**：``cv2.drawChessboardCorners`` 在点数少于
    ``patternSize`` 时**不报错**，而是直接**越界读内存**（实测：传 80 个点、
    patternSize=11x8 时会「成功返回」，画出来的是垃圾）。这会引发
    glibc 堆损坏 —— 本项目已经因为堆问题崩过好几次，所以这种"静默越界"
    的调用点必须一律加长度检查，宁可少画一张图。
    同时统一转成 ``float32``：``float64`` 不会报错但会被当成乱码解释
    （实测同一批角点 float32/float64 画出的像素量不同）。
    """
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
    if det.found and det.image_points is not None:
        want = int(np.prod(spec.pattern_size))
        pts = np.asarray(det.image_points, dtype=np.float32).reshape(-1, 1, 2)
        if len(pts) == want:
            cv2.drawChessboardCorners(
                img, tuple(int(v) for v in spec.pattern_size), pts, True)
    return img


def main() -> int:
    """CLI：对单张图/一个目录做检测，打印结果并可选导出可视化。"""
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="标定板检测自检")
    ap.add_argument("inputs", nargs="+", help="图像文件或目录")
    ap.add_argument("--pattern", default="9x6", help="内角点数，如 9x6")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--type", default="chessboard", choices=["chessboard", "charuco"])
    ap.add_argument("--dict", default="DICT_5X5_100")
    ap.add_argument("--marker-mm", type=float, default=0.0)
    ap.add_argument("--save-dir", default=None, help="把可视化结果存到这里")
    args = ap.parse_args()

    c, r = (int(v) for v in args.pattern.lower().split("x"))
    spec = BoardSpec(args.type, (c, r), args.square_mm / 1000.0,
                     args.marker_mm / 1000.0, args.dict)
    print(f"标定板: {spec}")
    print(f"3D 点数: {len(spec.object_points())}")

    files: list[Path] = []
    for s in args.inputs:
        p = Path(s)
        files += sorted(p.glob("*.png")) + sorted(p.glob("*.jpg")) if p.is_dir() else [p]

    outdir = Path(args.save_dir) if args.save_dir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for f in files:
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  ❌ 读不了 {f}")
            continue
        det = detect(gray, spec)
        if det.found:
            n_ok += 1
            pts = det.image_points.reshape(-1, 2)
            print(f"  ✅ {f.name:24s} {len(pts):3d} 点 方法={det.method:17s} "
                  f"x∈[{pts[:,0].min():6.1f},{pts[:,0].max():6.1f}] "
                  f"y∈[{pts[:,1].min():6.1f},{pts[:,1].max():6.1f}]")
            if outdir:
                cv2.imwrite(str(outdir / f"{f.stem}_det.png"),
                            draw_detection(gray, det, spec))
        else:
            print(f"  ❌ {f.name:24s} 未检出（方法={det.method}）")
    print(f"\n检出 {n_ok}/{len(files)}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
