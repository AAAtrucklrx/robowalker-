#!/usr/bin/env python3
"""AprilGrid 靶标检测（EuRoC 用的就是它）。

为什么需要这个
==================================================================
我们的 `target_detect` 支持棋盘格和 ChArUco，但**公开的相机-IMU 数据集
几乎都用 AprilGrid**（EuRoC、TUM VI、Kalibr 样例）。原因是 AprilGrid 对
**运动模糊和部分遮挡鲁棒得多** —— 每个 tag 有唯一编号，丢几个不影响其余，
而棋盘格一旦被遮挡或不完整就整块废掉。

所以想用真实相机-IMU 数据验证 C2/C3，就必须支持 AprilGrid。

AprilGrid 是什么
==================================================================
在平面上一格格摆开的 AprilTag（EuRoC 用 6x6 个 36h11 家族，tag 边长
0.088 m，间距 = 0.3 × 边长）。检测流程：

1. `cv2.aruco` 检出每个 tag 的 **4 个角点** 和它的 **唯一 ID**；
2. 由 ID 查出这个 tag 在网格里的**行列位置** → 算出它 4 个角点的
   **3D 坐标**（在靶标坐标系下）；
3. 于是每帧最多得到 **36 tag × 4 = 144 对** (3D ↔ 像素) 对应点，
   比棋盘格的 88 个还多。

因此**下游完全不用改** —— 拿到的还是 `Detection`（object_points +
image_points），和棋盘格一模一样，直接喂 PnP / 标定。

坐标系约定
==================================================================
靶标系：原点在网格左下角 tag 的左下角，X 向右，Y 向上，Z 垂直靶面向外。
（这个"约定"是任意的 —— 标定只关心**一致性**，靶标系随便定都不影响
手眼解，因为常量板系偏置会在 AX=XB 里抵消。）

ID → 行列的映射
==================================================================
用 Kalibr 的 AprilGrid 约定：**从左上角开始、行优先（向右递增 1，
换行向下递增 tagCols）**。

⚠️ 这个映射如果搞错，PnP 的重投影残差会立刻爆炸（因为角点被安到了
错误的位置）。所以 :func:`verify_layout` 会**用重投影残差反过来验证**
这个假设 —— 残差小说明映射对，残差大说明映射错。不靠记忆，靠数据。

用法::

    from aprilgrid import AprilGridSpec, detect_aprilgrid
    spec = AprilGridSpec(tag_cols=6, tag_rows=6, tag_size=0.088, spacing=0.3)
    det = detect_aprilgrid(gray, spec)
    print(det.found, len(det.image_points))
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from target_detect import Detection

# EuRoC / Kalibr 的 april_6x6.yaml 参数
EUROC_APRILGRID = dict(tag_cols=6, tag_rows=6, tag_size=0.088, spacing=0.3)


@dataclass
class AprilGridSpec:
    """AprilGrid 靶标规格。"""
    tag_cols: int = 6
    tag_rows: int = 6
    tag_size: float = 0.088        # tag 边长（米）
    spacing: float = 0.3           # 间距 / tag_size
    family: str = "DICT_APRILTAG_36h11"
    # ID→行列的映射约定。**不靠记忆，靠数据选**（见 verify_layout）：
    #   "row_major_tl": id = row*cols + col，row 0 是最上一行（左上角是 id 0）
    #   "col_major_bl": id = col*rows + row，row 0 是最下一行（左下角是 id 0）
    # 两种都是真实项目里用过的写法，谁对由重投影残差判定。
    layout: str = "row_major_tl"

    @property
    def pitch(self) -> float:
        """相邻 tag 中心之间的距离（米）。"""
        return self.tag_size * (1.0 + self.spacing)

    def tag_center(self, tid: int) -> tuple[float, float]:
        """tag ID → 它在靶标系里的中心坐标 (x, y)。

        Kalibr 约定：**左上角是 id 0，行优先**（向右 id+1，换行 id+tag_cols）。

        允许 id 超出范围时抛错 —— 宁可报错也不要静默算错坐标。
        """
        n = self.tag_cols * self.tag_rows
        if not (0 <= tid < n):
            raise ValueError(f"tag id {tid} 超出 0..{n-1}")
        if self.layout == "row_major_tl":
            row = tid // self.tag_cols          # 0 = 最上面一行
            col = tid % self.tag_cols
        elif self.layout == "col_major_bl":
            col = tid // self.tag_rows          # 0 = 最左一列
            row = self.tag_rows - 1 - (tid % self.tag_rows)   # 0 = 最上一行
        else:
            raise ValueError(f"未知 layout {self.layout!r}")
        # Y 轴向上 → 最上面一行 y 最大。+2.0 让原点落在网格外的角上，
        # 避免靶标点出现 0 坐标（对某些数值方法更友好）。
        x = (col + 2.0) * self.pitch
        y = (self.tag_rows - 1 - row + 2.0) * self.pitch
        return x, y

    def tag_object_points(self, tid: int) -> np.ndarray:
        """某个 tag 的 4 个角点在该 tag 自身坐标系下的 3D 坐标 (4,3)。

        `cv2.aruco` 返回的角点顺序是 **左上、右上、右下、左下**（按 tag 在
        图像里的朝向）。这里与之对应（Y 轴向上，所以"上"是 +y）。
        """
        s = self.tag_size / 2.0
        return np.array([
            [-s, +s, 0.0],      # 左上
            [+s, +s, 0.0],      # 右上
            [+s, -s, 0.0],      # 右下
            [-s, -s, 0.0],      # 左下
        ], dtype=np.float64)


def _dict(spec: AprilGridSpec):
    import cv2

    a = cv2.aruco
    if not hasattr(a, spec.family):
        raise ValueError(f"OpenCV 没有字典 {spec.family}")
    getter = getattr(a, "getPredefinedDictionary", None)
    d = getattr(a, spec.family)
    return getter(d) if getter else a.Dictionary_get(d)


# 官方 apriltag 库的检测器缓存（构造很贵，别每帧新建）
_AT_DETECTORS: dict = {}


def _official_apriltag(gray: np.ndarray, spec: AprilGridSpec):
    """用**官方 apriltag 库**检测，返回 [(id, 4x2 corners), ...] 或 None。

    为什么必须用它而不是 OpenCV
    ------------------------------------------------------------------
    实测（本文件作者踩了整整一轮）：

    * OpenCV 的 ``DICT_APRILTAG_36h11`` **检不出 Kalibr 的 AprilGrid 靶标** ——
      在一张**完全干净的靶标渲染图**上检出 **0** 个；而用 OpenCV 自己
      ``generateImageMarker`` 生成的 36h11 标记，自己检自己 **6/6 全对**。
      ⇒ OpenCV 的检测器没坏，是它和 Kalibr 靶标的**渲染方式**不兼容。
    * ``DetectorParameters.detectInvertedMarker`` **对 AprilTag 字典无效** ——
      把 OpenCV 生成的标记反色后，开不开这个开关都检不出（只对 ArUco 生效）。
    * 官方 ``apriltag`` 库（Kalibr 用的就是它）在实拍图上**正常**：
      帧 1023 检出 7 个，**ID 唯一无重复**（OpenCV 的误检则会给出重复 ID）。

    所以：AprilGrid 一律走官方库；检测不到再退回 OpenCV（聊胜于无，
    但要注意它的结果可能含重复 ID 的误检，用之前必须查重）。
    """
    try:
        import apriltag
    except ImportError:
        return None
    fam = spec.family.replace("DICT_APRILTAG_", "tag").lower()
    key = (fam, 1.5, 0.8)
    d = _AT_DETECTORS.get(key)
    if d is None:
        try:
            d = apriltag.Detector(apriltag.DetectorOptions(
                families=fam, quad_decimate=1.5, quad_blur=0.8,
                refine_edges=1))
        except Exception:                                  # noqa: BLE001
            return None
        _AT_DETECTORS[key] = d
    out = []
    for r in d.detect(gray):
        out.append((int(r.tag_id), np.asarray(r.corners, dtype=np.float64)))
    return out


def detect_aprilgrid(gray: np.ndarray, spec: AprilGridSpec,
                     corner_refine: bool = True) -> Detection:
    """检测 AprilGrid，返回与棋盘格同构的 :class:`Detection`。

    优先用**官方 apriltag 库**（见 :func:`_official_apriltag` 的说明），
    不可用时退回 OpenCV 的 aruco。
    """
    import cv2

    hits = _official_apriltag(gray, spec)
    if hits is not None:
        img_pts, obj_pts, used = [], [], []
        for tid, c in hits:
            try:
                cx, cy = spec.tag_center(tid)
            except ValueError:
                continue                     # id 超出网格 → 误检
            img_pts.append(c.astype(np.float32).reshape(-1, 2))
            obj_pts.append(spec.tag_object_points(tid)
                           + np.array([cx, cy, 0.0]))
            used.append(tid)
        if not img_pts:
            return Detection(False, None, None, method="apriltag")
        img = np.concatenate(img_pts, 0).astype(np.float32).reshape(-1, 1, 2)
        obj = np.concatenate(obj_pts, 0).astype(np.float32)
        return Detection(True, img, obj, method="apriltag",
                         marker_ids=np.asarray(used, dtype=np.int32))

    a = cv2.aruco
    d = _dict(spec)
    params = (a.DetectorParameters() if hasattr(a, "DetectorParameters")
              else a.DetectorParameters_create())
    # 靶标是印在哑光纸上的，允许一定模糊（EuRoC 运动快，图像常带模糊）
    try:
        params.adaptiveThreshWinSizeMin = 5
        params.adaptiveThreshWinSizeMax = 35
        params.cornerRefinementMethod = (a.CornerRefineMethod.SUBPIX
                                         if corner_refine else
                                         a.CornerRefineMethod.NONE)
    except Exception:                                    # noqa: BLE001
        pass

    if hasattr(a, "ArucoDetector"):
        det = a.ArucoDetector(d, params)
        corners, ids, _ = det.detectMarkers(gray)
    else:                                                # OpenCV < 4.7
        corners, ids, _ = a.detectMarkers(gray, d, parameters=params)

    if ids is None or len(ids) == 0:
        return Detection(False, None, None, method="aprilgrid")

    img_pts, obj_pts, used = [], [], []
    for c, tid in zip(corners, ids.ravel()):
        tid = int(tid)
        try:
            cx, cy = spec.tag_center(tid)
        except ValueError:
            continue                      # id 不在网格范围 → 误检，丢掉
        local = spec.tag_object_points(tid)
        local = local + np.array([cx, cy, 0.0])
        img_pts.append(np.asarray(c, dtype=np.float32).reshape(-1, 2))
        obj_pts.append(local)
        used.append(tid)

    if not img_pts:
        return Detection(False, None, None, method="aprilgrid")

    img = np.concatenate(img_pts, axis=0).astype(np.float32).reshape(-1, 1, 2)
    obj = np.concatenate(obj_pts, axis=0).astype(np.float32)
    return Detection(True, img, obj, method="aprilgrid",
                     marker_ids=np.asarray(used, dtype=np.int32))


def verify_layout(gray_paths, spec: AprilGridSpec, K=None, dist=None,
                  max_frames: int = 20, try_all: bool = True) -> dict:
    """**用重投影残差验证 ID→行列映射对不对。**

    这是本模块唯一的"自检"。做法：对若干帧做 PnP，看投影回图像的误差。
    * 残差 ~0.1 px → 映射对（3D 点和像素点配上了）
    * 残差几十 px   → 映射错（角点被安到了错误的位置）

    这样就不必依赖"我记得 Kalibr 是这么排的"，而是让数据说话。
    """
    import cv2

    if try_all:
        # 把候选 layout 都试一遍，让**重投影残差**来决定哪个对。
        # 这样就不必依赖"我记得 Kalibr 是这么排的"。
        cands = ["row_major_tl", "col_major_bl"]
        scored = []
        for lay in cands:
            sp = AprilGridSpec(spec.tag_cols, spec.tag_rows, spec.tag_size,
                               spec.spacing, spec.family, layout=lay)
            r = verify_layout(gray_paths, sp, K, dist, max_frames,
                              try_all=False)
            if r["rms_px"] is not None:
                scored.append((r["rms_px"], lay, r))
        if not scored:
            return {"n_frames": 0, "rms_px": None, "tags_per_frame": 0.0,
                    "tried": []}
        scored.sort(key=lambda x: x[0])
        best_rms, best_lay, best = scored[0]
        gap = (scored[1][0] / best_rms) if len(scored) > 1 and best_rms > 0 \
            else float("inf")
        return {"n_frames": best["n_frames"], "rms_px": best_rms,
                "tags_per_frame": best["tags_per_frame"],
                "layout": best_lay,
                "tried": [(lay, rms) for rms, lay, _ in scored],
                # 两个候选差得够开 → 结论可信；接近 → 说明这个判据区分不开
                "decisive": gap > 3.0}

    res = []
    n_tag = []
    for p in list(gray_paths)[:max_frames]:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        det = detect_aprilgrid(g, spec)
        if not det.found or len(det.image_points) < 8:
            continue
        n_tag.append(len(det.image_points) // 4)
        if K is None:
            # 没给内参就用一个粗略的初值（只为了看"映射对不对"，
            # 不追求数值精度）；焦距取图像宽的 1.2 倍是常见量级。
            h, w = g.shape
            K0 = np.array([[w * 1.2, 0, w / 2],
                           [0, w * 1.2, h / 2], [0, 0, 1.0]])
            d0 = np.zeros(5)
        else:
            K0, d0 = np.asarray(K, float), np.asarray(dist, float)
        ok, rvec, tvec = cv2.solvePnP(det.object_points, det.image_points,
                                      K0, d0, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(det.object_points, rvec, tvec, K0, d0)
        e = proj.reshape(-1, 2) - det.image_points.reshape(-1, 2)
        res.append(float(np.sqrt((e ** 2).sum(axis=1).mean())))
    return {"n_frames": len(res), "rms_px": float(np.mean(res)) if res else None,
            "tags_per_frame": float(np.mean(n_tag)) if n_tag else 0.0}


if __name__ == "__main__":
    import argparse
    import glob as _glob
    import sys
    from pathlib import Path

    ap = argparse.ArgumentParser(description="AprilGrid 检测 + 布局验证")
    ap.add_argument("--dir", required=True, help="图片目录")
    ap.add_argument("--glob", default="*.png")
    ap.add_argument("--tag-cols", type=int, default=6)
    ap.add_argument("--tag-rows", type=int, default=6)
    ap.add_argument("--tag-size", type=float, default=0.088)
    ap.add_argument("--spacing", type=float, default=0.3)
    ap.add_argument("--check", action="store_true",
                    help="用重投影残差验证 ID→行列映射（**强烈建议先跑这个**）")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calib"))
    spec = AprilGridSpec(args.tag_cols, args.tag_rows, args.tag_size,
                         args.spacing)
    files = sorted(_glob.glob(str(Path(args.dir) / args.glob)))
    print(f"目录 {args.dir}：{len(files)} 张")
    print(f"靶标：{args.tag_cols}x{args.tag_rows} AprilTag，"
          f"边长 {args.tag_size} m，间距 {args.spacing}× "
          f"→ 中心距 {spec.pitch:.4f} m")

    n_ok = 0
    tags = []
    for p in files[:40]:
        import cv2

        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        d = detect_aprilgrid(g, spec)
        if d.found:
            n_ok += 1
            tags.append(len(d.image_points) // 4)
    print(f"检出 {n_ok}/{min(len(files),40)} 张，"
          f"平均 {np.mean(tags) if tags else 0:.1f} 个 tag/帧")

    if args.check:
        r = verify_layout(files, spec, max_frames=20)
        print(f"\n布局验证：{r['n_frames']} 帧，"
              f"平均 {r['tags_per_frame']:.1f} tag/帧")
        if r["rms_px"] is None:
            print("  ❌ 没有可用的帧")
        elif r["rms_px"] < 1.0:
            print(f"  ✅ 重投影 {r['rms_px']:.3f} px → "
                  f"ID→行列映射**正确**")
        else:
            print(f"  ❌ 重投影 {r['rms_px']:.1f} px → "
                  f"映射**可能错了**（角点被安到了错误位置）")
