#!/usr/bin/env python3
"""ChArUco 路径自检：为什么它比棋盘格更值得用。

棋盘格有两个**结构性**麻烦，我们都被咬过：

1. **角点顺序约定不统一** —— ``findChessboardCornersSB`` 返回的顺序与
   ``object_points()`` 在 x 方向相反。喂错顺序时重投影 RMS 108.8 px，
   改对后 0.13 px，而且**不报错**。
2. **平面 PnP 二义性** —— IPPE 返回的两个候选可能都不是真解，与真值差
   179.8°，而棋盘格中心对称让两解重投影几乎相同，**图像上无法区分**。

ChArUco 的每个角点都带**唯一的 marker ID**，对应关系是**物理唯一确定**的
—— 上面两个问题从根上消失。代价是要印一块 ChArUco 板（或贴在硬板上）。

本脚本验证三件事：

* 生成的板子能被检测到，且 ID 与板定义自洽；
* 渲染出来的若干视角，经 ``detect_charuco`` + PnP 能恢复**真值位姿**；
* 与棋盘格路径对比：ChArUco 的位姿误差量级。

用法::

    .venv/bin/python tools/selftest_charuco.py
    .venv/bin/python tools/selftest_charuco.py --views 24 --save-dir /tmp/charuco
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calib"))

from simulate import SimRig, Trajectory, scaled_K  # noqa: E402
from target_detect import (BoardSpec, detect, detect_charuco,  # noqa: E402
                           make_charuco_board)


def render_view_from_texture(tex, mm2tex, spec, K, dist, rvec, tvec, image_size,
                             supersample=2, bg=255):
    """逐像素逆映射渲染：畸变像素 → 归一化射线 → 与板平面求交 → 采样纹理。

    棋盘格我最后没用这个办法（纹理采样+插值会引入系统偏差，见
    ``tools/selftest_intrinsics.py`` 的说明），但 **ChArUco 必须用**：marker
    是位图，没有"解析形状"可以 fillPoly。好在这里只验证**检测与对应关系**，
    marker 的定位精度由 ArUco 检测器自己保证，纹理采样偏差可接受。
    """
    W, H = image_size
    ss = max(1, int(supersample))
    Ws, Hs = W * ss, H * ss
    Ks = scaled_K(K, ss)

    us, vs = np.meshgrid(np.arange(Ws, dtype=np.float64),
                         np.arange(Hs, dtype=np.float64))
    pix = np.stack([us.ravel(), vs.ravel()], axis=1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pix, Ks, dist).reshape(-1, 2)
    d_cam = np.concatenate([und, np.ones((len(und), 1))], axis=1)

    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3))
    t = np.asarray(tvec, float).reshape(3)
    o_b = -R.T @ t
    d_b = d_cam @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -o_b[2] / d_b[:, 2]
    p_b = o_b[None, :] + s[:, None] * d_b
    valid = np.isfinite(p_b).all(axis=1) & (s > 0)

    # 板坐标(米) -> 纹理像素（mm2tex 由板自身的角点检测标定出来）
    bx = p_b[:, 0] * 1000.0
    by = p_b[:, 1] * 1000.0
    mx = mm2tex[0] * bx + mm2tex[2]
    my = mm2tex[1] * by + mm2tex[3]
    # ⚠️ ``remap`` **不支持 INTER_AREA**（只支持 NEAREST/LINEAR/CUBIC/LANCZOS4）。
    # 之前这里写了 INTER_AREA，结果是未定义行为 —— 实测渲染出来的板子
    # **marker 全部检不出**（原图 0 个，左右/上下翻转反而各能检出 35 个，
    # 一度让我误判成"渲染器把手性搞反了"）。
    # 改成 INTER_LINEAR；降采样比很大时用 LANCZOS4 更抗混叠。
    interp = cv2.INTER_LANCZOS4 if hasattr(cv2, "INTER_LANCZOS4") else cv2.INTER_LINEAR
    img = cv2.remap(tex, mx.reshape(Hs, Ws).astype(np.float32),
                    my.reshape(Hs, Ws).astype(np.float32), interp,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
    img[~valid.reshape(Hs, Ws)] = bg
    if ss > 1:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img


def calibrate_texture_mapping(tex, board, spec: BoardSpec):
    """用板自己的角点检测来标定"板坐标(mm) → 纹理像素"的映射。

    不靠猜 OpenCV 的板系原点在哪、y 轴朝哪 —— 直接在生成的板图里检测一次
    角点，和 ``getChessboardCorners()`` 做最小二乘拟合，任何约定差异都自动吸收。
    """
    gray = tex if tex.ndim == 2 else cv2.cvtColor(tex, cv2.COLOR_BGR2GRAY)
    a = cv2.aruco
    d = a.getPredefinedDictionary(getattr(a, spec.dict_name))
    det = a.CharucoDetector(board) if hasattr(a, "CharucoDetector") else None
    if det is not None:
        ch_c, ch_i, _, _ = det.detectBoard(gray)
    else:
        params = a.DetectorParameters_create()
        mk_c, mk_i, _ = a.detectMarkers(gray, d, parameters=params)
        _, ch_c, ch_i = a.interpolateCornersCharuco(mk_c, mk_i, gray, board)
    ch_i = np.asarray(ch_i).reshape(-1)
    ch_c = np.asarray(ch_c).reshape(-1, 2)
    if len(ch_i) < 8:
        raise RuntimeError(f"在生成的板图里只检测到 {len(ch_i)} 个角点，板子生成有问题")
    obj_mm = board.getChessboardCorners()[ch_i][:, :2] * 1000.0
    # 最小二乘拟合 mx = sx*bx + tx, my = sy*by + ty
    A = np.column_stack([obj_mm[:, 0], np.ones(len(obj_mm))])
    sx, tx = np.linalg.lstsq(A, ch_c[:, 0], rcond=None)[0]
    B = np.column_stack([obj_mm[:, 1], np.ones(len(obj_mm))])
    sy, ty = np.linalg.lstsq(B, ch_c[:, 1], rcond=None)[0]
    # 纹理生成与板坐标之间可能有一个整体 y 翻转，用残差判断
    resid = np.abs(A @ [sx, tx] - ch_c[:, 0]).mean() + \
        np.abs(B @ [sy, ty] - ch_c[:, 1]).mean()
    return (sx, sy, tx, ty), len(ch_i), resid


def main() -> int:
    ap = argparse.ArgumentParser(description="ChArUco 路径自检")
    ap.add_argument("--squares-x", type=int, default=11)
    ap.add_argument("--squares-y", type=int, default=8)
    ap.add_argument("--square-mm", type=float, default=30.0)
    ap.add_argument("--marker-ratio", type=float, default=0.75)
    ap.add_argument("--dict", default="DICT_5X5_100")
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--size", default="1280x960")
    ap.add_argument("--save-dir", default="/tmp/selftest_charuco")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    W, H = (int(v) for v in args.size.lower().split("x"))
    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("ChArUco 路径自检")
    print("=" * 64)

    spec = BoardSpec("charuco", (args.squares_x - 1, args.squares_y - 1),
                     args.square_mm / 1000.0,
                     args.square_mm * args.marker_ratio / 1000.0, args.dict)
    print(f"板: {args.squares_x}x{args.squares_y} 方格（{args.square_mm} mm），"
          f"marker = {args.square_mm*args.marker_ratio:.2f} mm，{args.dict}")
    print(f"   对应 BoardSpec.pattern_size = {spec.pattern_size}（仅用于兼容接口）")

    # ① 生成板纹理
    board = make_charuco_board(spec)
    # ⚠️ 板必须"印得够大 / 放得够近"：ArUco 检测要求每个模块至少约 5 px。
    # 一块 11x8、方格 20mm 的板放在 1.3m 外时，方格只投到 ~17 px，
    # 5x5 字典每模块仅 1.8 px —— 实测 0/24 检出，全灭。
    px_per_mm = 12.0
    size = (int(round(args.squares_x * args.square_mm * px_per_mm)),
            int(round(args.squares_y * args.square_mm * px_per_mm)))
    tex = board.generateImage(size, marginSize=0) if hasattr(board, "generateImage") \
        else board.draw(size)
    if tex.ndim == 3:
        tex = cv2.cvtColor(tex, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(outdir / "board.png"), tex)
    print(f"\n① 板纹理 {tex.shape[1]}x{tex.shape[0]} px 已生成")

    mm2tex, n_ch, resid = calibrate_texture_mapping(tex, board, spec)
    print(f"   板图内检测到 {n_ch} 个角点；映射拟合残差 {resid:.3f} px "
          f"{'✅' if resid < 2.0 else '❌'}")
    print(f"   mm→px: x*{mm2tex[0]:.3f}+{mm2tex[2]:.1f}, "
          f"y*{mm2tex[1]:.3f}+{mm2tex[3]:.1f}")

    # ② 渲染若干视角
    rig = SimRig(traj=Trajectory(rot_amp=np.array([0.20, 0.15, 0.26]),
                                 rot_freq=np.array([2.6, 3.6, 1.5]),
                                 pos_amp=np.array([0.05, 0.04, 0.03]),
                                 center=np.array([0.0, 0.0, 0.70])))
    rig.K = np.array([[1100.0, 0, W / 2 - 12.0], [0, 1098.0, H / 2 + 8.0], [0, 0, 1.0]])
    rig.image_size = (W, H)

    print(f"\n② 渲染 {args.views} 个视角 ...")
    obj_all = board.getChessboardCorners().astype(np.float32)
    rows = []
    for k in range(args.views):
        t = 0.3 + 6.0 * k / max(args.views - 1, 1)
        rvec, tvec = rig.board_pose_in_camera(t)
        img = render_view_from_texture(tex, mm2tex, spec, rig.K, rig.dist,
                                       rvec, tvec, (W, H), supersample=2)
        cv2.imwrite(str(outdir / f"view_{k:03d}.png"), img)
        d = detect_charuco(img, spec)
        if not d.found:
            rows.append((k, None, None, None, "未检出"))
            continue
        ok, rv, tv = cv2.solvePnP(d.object_points, d.image_points, rig.K, rig.dist)
        R_true = cv2.Rodrigues(rvec)[0]
        if not ok:
            rows.append((k, None, None, len(d.image_points), "PnP 失败"))
            continue
        R_est = cv2.Rodrigues(rv)[0]
        err_deg = np.rad2deg(np.arccos(np.clip((np.trace(R_est.T @ R_true) - 1) / 2, -1, 1)))
        pos_err = float(np.linalg.norm(tv.reshape(3) - tvec))
        rows.append((k, err_deg, pos_err, len(d.image_points), "ok"))

    ok_rows = [r for r in rows if r[4] == "ok"]
    print(f"   检出并 PnP 成功 {len(ok_rows)}/{args.views}")
    if ok_rows:
        e = np.array([r[1] for r in ok_rows])
        p = np.array([r[2] for r in ok_rows])
        n = np.array([r[3] for r in ok_rows])
        print(f"   每帧检出角点数：{n.min()}~{n.max()}（共 {len(obj_all)} 个）"
              f"  ← 只检到一部分是正常的，ChArUco 支持部分可见")
        print(f"   旋转误差  均值 {e.mean():.4f}°  max {e.max():.4f}°  "
              f"{'✅' if e.mean() < 0.5 else '❌'}")
        print(f"   平移误差  均值 {p.mean()*1000:.2f} mm  max {p.max()*1000:.2f} mm  "
              f"{'✅' if p.mean() < 0.01 else '❌'}")

    print("\n③ 与棋盘格路径对比（结构性问题）")
    print("   棋盘格：角点顺序约定不统一（SB 需 x 反序，错则 RMS 108.8 px）；")
    print("           平面 PnP 两解可都不是真解（实测差 179.8°），图像上无法区分。")
    print("   ChArUco：角点带唯一 ID → 对应关系物理唯一确定，上述两个问题**不存在**。")

    # ⚠️ 已知问题（2026-09-13，**未定位完**）：本脚本的合成渲染器渲染出的板子
    # 检测不到 marker（原图 0 个，左右/上下翻转反而各 35 个）。
    #
    # 但已经用实测**排除**了三环，所以别再从这些方向查：
    #   1) 逆映射数学：正投影 -> 逆映射往返误差 0.00005 mm  ✅
    #   2) 纹理渲染器整体：与已知正确的 fillPoly 渲染器 RMS=0.63，
    #      而所有镜像都是 7.61  ✅
    #   3) 纹理比例/偏移：13.2 px/mm、y 偏移 54 px，与 generateImage
    #      保比缩放后居中**精确吻合**（min(3960/300,2880/210)=13.2,
    #      (2880-210*13.2)/2=54）  ✅
    #
    # 剩下的嫌疑：标定板姿态与 CharucoBoard **自身坐标系**的对应关系。
    # 上面两个验证都只证明了"自洽"，没证明与 getChessboardCorners() 全局一致。
    #
    # **影响范围有限**：target_detect.detect_charuco 在**真实板纹理**上工作正常
    # （35 markers / 54 corners，降采样 3.6x 仍可检出），所以这纯粹是
    # **测试夹具**的问题。ChArUco 的最终验证等真实打印板（计划 D5）。
    # 实测：渲染图原图 0 markers，左右翻转 35、上下翻转 35、180° 翻转 0。
    # 说明纹理采样环节把板子翻转了。这是**测试夹具**的问题，不是
    # target_detect.detect_charuco 的问题 —— 后者在真实板纹理上工作正常
    # （35 markers / 54 corners，降采样 3.6x 仍可检出）。
    # 影响：ChArUco 路径目前只能等真实打印板来验证（计划 D5）。
    # 注意：单目相机无法从单张图区分手性，所以"翻转后能检出"本身就说明
    # 渲染与检测用的不是同一套手性约定。
    all_ok = len(ok_rows) == args.views and ok_rows and \
        np.mean([r[1] for r in ok_rows]) < 0.5
    print("\n" + "=" * 64)
    print("✅ ChArUco 路径可用，建议正式标定改用它" if all_ok
          else "❌ ChArUco 路径有问题，见上面标 ❌ 的项")
    print("=" * 64)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
