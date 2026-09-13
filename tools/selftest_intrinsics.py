#!/usr/bin/env python3
"""C1 链路自检：**两层验证**，不用实物标定板也能确认代码是对的。

为什么要分两层
------------------------------------------------------------------
验证的精度**不可能超过测试夹具本身的精度**。所以把"标定数学"和
"图像检测"分开测，各自用匹配的容差：

**Level 1 · 点级（纯数学）**
    先用真值 K/dist 把角点**正向投影**成 2D 点，直接喂给标定器。
    这一步完全不经过图像，可以把容差卡到 0.1 px 量级。
    它回答的是："我的标定/验证/误差计算写对了吗？"

**Level 2 · 图像级（端到端）**
    把棋盘格**渲染**成图像，再走 findChessboardCorners → 标定。
    它额外检验检测环节，但精度上限受制于渲染器的边缘模型
    （实测残差地板约 0.08 px），所以容差必须放宽。
    它回答的是："拿真实照片进来，整条链路能跑通吗？"

> 实测数据（本脚本 ss=4，36→200 个位姿）：
> ``cx/cy`` 的偏差从 0.89 px 收敛到 0.10 px（有限样本方差），
> 但 ``fx/fy`` 稳定偏 +0.8 px、残差稳定在 0.08 px，**与位姿数量无关**
> → 这是渲染器（多边形填充 + 超采样）的边缘量化，不是标定算法的问题。
> 这个区分很重要：不做收敛性实验，很容易把夹具误差误判成算法 bug。

比容差数字更有意义的畸变检查
------------------------------------------------------------------
径向畸变系数 ``k1/k2/k3`` 之间**高度相关**，单独比系数意义不大
（k1 偏 0.02 而 k2 反向偏 0.02，物理上可能完全等价）。所以本脚本
额外比较**畸变函数本身**：在若干归一化半径处比较

    1 + k1 r^2 + k2 r^4 + k3 r^6

只要这个函数对得上，畸变就是对的，系数怎么分配并不重要。

用法::

    .venv/bin/python tools/selftest_intrinsics.py                    # 两层都跑
    .venv/bin/python tools/selftest_intrinsics.py --mode points      # 只跑纯数学
    .venv/bin/python tools/selftest_intrinsics.py --noise-px 0.3     # 点级加像素噪声
    .venv/bin/python tools/selftest_intrinsics.py --mode images --save-dir /tmp/syn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calib"))

from intrinsics import calibrate, reprojection_rms, validate_on  # noqa: E402
from target_detect import BoardSpec, Detection, detect  # noqa: E402


# ══════════════════════════════════════════════════════════════
#  真值
# ══════════════════════════════════════════════════════════════
def make_truth(image_size):
    W, H = image_size
    fx, fy = 1100.0, 1098.0
    cx, cy = W / 2 - 12.0, H / 2 + 8.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    dist = np.array([-0.12, 0.045, 0.0012, -0.0008, 0.0], float)
    return K, dist


def radial_factor(dist, r2):
    k1, k2, p1, p2, k3 = (list(dist) + [0] * 5)[:5]
    return 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3


def board_outline(spec: BoardSpec) -> np.ndarray:
    cols, rows = spec.pattern_size
    s = spec.square_size
    return np.array([[-s, -s, 0], [cols * s, -s, 0],
                     [cols * s, rows * s, 0], [-s, rows * s, 0]], np.float32)


def random_pose(rng, spec, K, dist, image_size, depth_range=(0.45, 1.10),
                max_tilt_deg=30.0, margin_frac=0.05, max_try=300):
    """随机位姿，保证整块板的 4 个外角都投影在画面内。

    让自检的变量可控：检测不到就等于在测"检测器的鲁棒性"，
    而这里要测的是"标定链路对不对"。真实采集时同样应让整板入画。
    """
    W, H = image_size
    cols, rows = spec.pattern_size
    s = spec.square_size
    center_b = np.array([(cols - 1) * s / 2, (rows - 1) * s / 2, 0.0])
    outl = board_outline(spec)
    mx, my = margin_frac * W, margin_frac * H
    for _ in range(max_try):
        lim = 0.28
        xn, yn = rng.uniform(-lim, lim), rng.uniform(-lim, lim)
        Z = rng.uniform(*depth_range)
        c_cam = np.array([xn * Z, yn * Z, Z])
        ang = np.deg2rad(max_tilt_deg)
        rvec = rng.uniform(-ang, ang, 3)
        rvec[2] *= 2.0
        R, _ = cv2.Rodrigues(rvec)
        tvec = c_cam - R @ center_b
        p = cv2.projectPoints(outl, rvec, tvec, K, dist)[0].reshape(-1, 2)
        if (p[:, 0].min() > mx and p[:, 0].max() < W - mx and
                p[:, 1].min() > my and p[:, 1].max() < H - my):
            return rvec, tvec
    raise RuntimeError("采样不出整板入画的位姿")


# ══════════════════════════════════════════════════════════════
#  渲染（Level 2 用）
# ══════════════════════════════════════════════════════════════
def scaled_K(K, ss):
    """超采样 ss 倍后的内参。原图 (u,v) 对应 ss 倍图的 (ss*u+(ss-1)/2, ...)。"""
    Ks = np.asarray(K, float).copy()
    Ks[0, 0] *= ss
    Ks[1, 1] *= ss
    Ks[0, 2] = K[0, 2] * ss + (ss - 1) / 2.0
    Ks[1, 2] = K[1, 2] * ss + (ss - 1) / 2.0
    return Ks


def render_view(spec, K, dist, rvec, tvec, image_size, supersample=4,
                bg=255, noise_sigma=0.0, rng=None):
    """渲染一帧棋盘格：逐黑方块正向投影 + fillPoly，再降采样抗锯齿。"""
    W, H = image_size
    ss = max(1, int(supersample))
    Ks = scaled_K(K, ss)
    img = np.full((H * ss, W * ss), bg, np.uint8)

    ncols, nrows = spec.pattern_size[0] + 1, spec.pattern_size[1] + 1
    s = spec.square_size
    quads = []
    for r in range(nrows):
        for c in range(ncols):
            if (r + c) % 2:                      # 只画黑格
                continue
            # 板系原点在第一个内角点，故物理板面左上角在 (-s, -s)
            x0, y0 = (c - 1) * s, (r - 1) * s
            quads.append([[x0, y0, 0], [x0 + s, y0, 0],
                          [x0 + s, y0 + s, 0], [x0, y0 + s, 0]])
    proj, _ = cv2.projectPoints(np.asarray(quads, np.float32).reshape(-1, 3),
                                rvec, tvec, Ks, dist)
    for q in proj.reshape(-1, 4, 2):
        cv2.fillPoly(img, [np.round(q).astype(np.int32)], 0)

    if ss > 1:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    if noise_sigma > 0:
        g = rng if rng is not None else np.random
        img = np.clip(img.astype(np.float32)
                      + g.normal(0, noise_sigma, img.shape), 0, 255).astype(np.uint8)
    return img


# ══════════════════════════════════════════════════════════════
#  比较与报告
# ══════════════════════════════════════════════════════════════
def compare_rows(K_true, dist_true, est, tol_k, label):
    """打印参数比对表，返回 (是否全过, 行数)。"""
    ok = True
    print(f"\n  {'参数':>6} {'真值':>12} {'解出':>12} {'误差':>10} {'容差':>8}")
    for name, tv, ev, tol in (("fx", K_true[0, 0], est.fx, tol_k[0]),
                              ("fy", K_true[1, 1], est.fy, tol_k[1]),
                              ("cx", K_true[0, 2], est.cx, tol_k[2]),
                              ("cy", K_true[1, 2], est.cy, tol_k[3])):
        err, good = ev - tv, abs(ev - tv) <= tol
        ok &= good
        print(f"  {name:>6} {tv:12.3f} {ev:12.3f} {err:+10.3f} {tol:8.2f} "
              f"{'✅' if good else '❌'}")

    print(f"\n  径向畸变函数 1+k1r²+k2r⁴+k3r⁶（比系数更有物理意义）")
    print(f"  {'r':>6} {'真值':>12} {'解出':>12} {'相对误差':>12}")
    for r in (0.2, 0.4, 0.6, 0.8):
        f_t = radial_factor(dist_true, r * r)
        f_e = radial_factor(est.dist, r * r)
        rel = abs(f_e - f_t) / abs(f_t)
        good = rel < tol_k[4]
        ok &= good
        print(f"  {r:6.1f} {f_t:12.6f} {f_e:12.6f} {rel*100:11.3f}% "
              f"{'✅' if good else '❌'}")

    for i, (tv, ev) in enumerate(zip(dist_true, est.dist)):
        tol = tol_k[5 + i] if len(tol_k) > 5 + i else 0.1
        good = abs(ev - tv) <= tol
        print(f"  参考 d{i+1} {tv:11.6f} {ev:12.6f} {ev-tv:+10.6f} {tol:8.4f} "
              f"{'✅' if good else '❌'}")
    return ok


def level1_points(spec, image_size, n_views, rng, noise_px):
    """Level 1：正向投影出角点，直接标定。不经过图像。"""
    print("\n" + "─" * 62)
    print(f"Level 1 · 点级（纯数学）  {n_views} 个位姿，角点噪声 {noise_px} px")
    print("─" * 62)
    K_true, dist_true = make_truth(image_size)
    obj = spec.object_points()
    dets = []
    for _ in range(n_views):
        rvec, tvec = random_pose(rng, spec, K_true, dist_true, image_size)
        p = cv2.projectPoints(obj, rvec, tvec, K_true, dist_true)[0]
        p = p.reshape(-1, 1, 2).astype(np.float32)
        if noise_px > 0:
            p = (p + rng.normal(0, noise_px, p.shape)).astype(np.float32)
        dets.append(Detection(True, p, obj, "projected"))

    est = calibrate(dets, image_size)
    res = _residuals(est, dets)
    print(f"  标定残差 RMS = {res['rms_px']:.3e} px")
    # 无噪声时应该接近机器精度；加点噪声也应在噪声量级
    floor = 1e-4 if noise_px == 0 else noise_px * 0.5
    resid_ok = res["rms_px"] < floor
    print(f"  残差阈值 {floor:.2e} px  {'✅' if resid_ok else '❌'}")

    # 点级可以卡得很紧：没有检测环节的误差
    tol_k = (0.05, 0.05, 0.5, 0.5, 0.002, 0.002, 0.004, 0.0005, 0.0005, 0.01)
    ok = compare_rows(K_true, dist_true, est, tol_k, "points") and resid_ok
    print(f"\n  Level 1 结论：{'✅ 通过（标定数学正确）' if ok else '❌ 未通过'}")
    return ok, K_true, dist_true


def level2_images(spec, image_size, n_views, rng, noise_sigma, supersample, outdir):
    """Level 2：渲染图像 → 检测 → 标定。"""
    print("\n" + "─" * 62)
    print(f"Level 2 · 图像级（端到端）  {n_views} 张渲染图，"
          f"图像噪声 {noise_sigma}，超采样 {supersample}x")
    print("─" * 62)
    K_true, dist_true = make_truth(image_size)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("syn_*.png"):
        old.unlink()
    for i in range(n_views):
        rvec, tvec = random_pose(rng, spec, K_true, dist_true, image_size)
        cv2.imwrite(str(outdir / f"syn_{i:04d}.png"),
                    render_view(spec, K_true, dist_true, rvec, tvec, image_size,
                                supersample=supersample, noise_sigma=noise_sigma,
                                rng=rng))

    dets = []
    for p in sorted(outdir.glob("syn_*.png")):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        dets.append(detect(gray, spec))
    n_ok = sum(1 for d in dets if d.found)
    print(f"  检出 {n_ok}/{len(dets)} 张（应全部检出；漏检说明渲染或检测有问题）")
    if n_ok < len(dets):
        print("  ❌ 有漏检")
        return False

    est = calibrate(dets, image_size)
    res = _residuals(est, dets)
    print(f"  标定残差 RMS = {res['rms_px']:.5f} px（渲染器边缘模型的地板 ≈0.08 px）")

    # 图像级容差必须放宽：验证精度不可能超过夹具精度。
    # 关键是"随数据量收敛"（脚本 docstring 里的实验），而不是绝对值很小。
    tol_k = (2.0, 2.0, 2.0, 2.0, 0.01, 0.02, 0.05, 0.003, 0.003, 0.15)
    ok = compare_rows(K_true, dist_true, est, tol_k, "images")
    resid_ok = res["rms_px"] < 0.15
    ok &= resid_ok
    print(f"\n  残差阈值 0.15 px  {'✅' if resid_ok else '❌'}")
    print(f"\n  Level 2 结论：{'✅ 通过（检测+标定端到端可用）' if ok else '❌ 未通过'}")
    return ok


def _residuals(intr, dets):
    vals = []
    for d in dets:
        ok, rvec, tvec = cv2.solvePnP(d.object_points, d.image_points,
                                      intr.K, intr.dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            vals.append(reprojection_rms(d.object_points, d.image_points,
                                         rvec, tvec, intr.K, intr.dist))
    a = np.asarray(vals)
    return {"n": len(vals), "rms_px": float(np.sqrt((a ** 2).mean())),
            "max_px": float(a.max())}


# ══════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description="C1 两层自检")
    ap.add_argument("--mode", choices=["points", "images", "both"], default="both")
    ap.add_argument("--views", type=int, default=40, help="Level 1 位姿数 / Level 2 图数")
    ap.add_argument("--pattern", default="9x6")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--size", default="1280x960")
    ap.add_argument("--noise-px", type=float, default=0.0, help="Level 1 角点噪声 (px)")
    ap.add_argument("--noise", type=float, default=0.0, help="Level 2 图像噪声 sigma")
    ap.add_argument("--supersample", type=int, default=4)
    ap.add_argument("--save-dir", default="/tmp/selftest_intrinsics")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    W, H = (int(v) for v in args.size.lower().split("x"))
    c, r = (int(v) for v in args.pattern.lower().split("x"))
    spec = BoardSpec("chessboard", (c, r), args.square_mm / 1000.0)

    print("=" * 62)
    print("Camera-IMU 标定 · C1 自检（合成数据，不需要实物标定板）")
    print("=" * 62)
    print(f"分辨率 {W}x{H}   标定板 {spec}")

    results = {}
    if args.mode in ("points", "both"):
        results["Level 1 点级"], _, _ = level1_points(
            spec, (W, H), args.views, rng, args.noise_px)
    if args.mode in ("images", "both"):
        results["Level 2 图像级"] = level2_images(
            spec, (W, H), args.views, rng, args.noise, args.supersample,
            args.save_dir)

    print("\n" + "=" * 62)
    for k, v in results.items():
        print(f"  {k:16s} {'✅ 通过' if v else '❌ 未通过'}")
    allok = all(results.values())
    print("=" * 62)
    print("✅ C1 链路可信，可以开始用真实数据了" if allok
          else "❌ 有未通过项，先修代码再采集真实数据")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
