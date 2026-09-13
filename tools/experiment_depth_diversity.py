#!/usr/bin/env python3
"""受控实验：**深度多样性对内参可辨识性的影响**。

这个实验的由来
------------------------------------------------------------------
"与成熟工具对比"（ROS ``camera_calibration``）时发现：**同一批图**，
我们的 ``fx=1135.7``、ROS ``fx=1099.5``，而真值是 ``1100.0``。
两者**重投影误差都是 0.09 px** —— 光看残差完全发现不了谁对谁错。

于是做了这个受控实验：**锁死真值 ``K`` 不变，只改相机相对标定板的深度跨度**，
看解出来的 ``fx`` 偏多少。

实测结论（真值 fx=1100, k1=-0.12, k2=0.045, k3=0）
------------------------------------------------------------------
| 深度跨度 | 解出 fx | fx 误差 | k3（真值 0） | 重投影 RMS |
|---|---|---|---|---|
| 1.0x（1.27~1.33 m） | 1145.9 | **+4.17%** | -0.047 | 0.128 px |
| 1.8x（0.85~1.55 m） | 1098.7 | -0.12% | -0.013 | 0.149 px |
| 3.3x（0.60~2.00 m） | 1103.5 | +0.31% | +0.004 | 0.153 px |

**三行的重投影误差几乎一样（0.13~0.15 px），但 fx 误差差了一个数量级。**

为什么
------------------------------------------------------------------
相机几乎只在同一距离上"转"，那么"物体看起来小"既可以是**焦距小**、
也可以是**离得远**，两者对成像的影响几乎等价 —— ``fx`` 与径向畸变
（尤其 ``k3``）形成**简并方向**。优化器在这个浅谷里随便落一个点，
残差都很小，但物理参数是错的。

**这条对采集协议的指导意义（比 bug 更重要）**：

> 标定时标定板必须走遍**近 / 中 / 远三档距离**，深度跨度至少 **2×**。
> 只在同一个距离上换角度，内参（尤其 fx 与畸变）是不可信的 ——
> 而且**重投影误差不会告诉你**。

用法::

    .venv/bin/python tools/experiment_depth_diversity.py
    .venv/bin/python tools/experiment_depth_diversity.py --views 80 --rate 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calib"))

from intrinsics import calibrate  # noqa: E402
from simulate import SimRig, Trajectory  # noqa: E402
from target_detect import BoardSpec, detect  # noqa: E402

TRUE = {"fx": 1100.0, "fy": 1098.0, "cx": 628.0, "cy": 488.0,
        "k1": -0.12, "k2": 0.045, "k3": 0.0}

CASES = [
    ("窄深度（相机几乎不动）", (0.05, 0.04, 0.03), 1.30),
    ("中深度", (0.05, 0.04, 0.35), 1.20),
    ("宽深度（推荐）", (0.06, 0.05, 0.70), 1.30),
]


def run_case(label, pos_amp, center_z, spec, dur, rate, seed, save_dir=None):
    rng = np.random.default_rng(seed)
    rig = SimRig(traj=Trajectory(rot_amp=np.array([0.16, 0.13, 0.22]),
                                 rot_freq=np.array([2.6, 3.6, 1.5]),
                                 pos_amp=np.array(pos_amp),
                                 pos_freq=np.array([0.8, 1.2, 0.45]),
                                 center=np.array([0.0, 0.0, center_z]),
                                 lead_in_s=0.0))
    t, imgs, _, _ = rig.make_camera_stream(0.0, dur, rate=rate, board_spec=spec,
                                           supersample=2, noise_sigma=0.0,
                                           rng=rng)
    zs = np.array([rig.traj.p_WC(x)[2] for x in t])
    dets = []
    for im in imgs:
        d = detect(im, spec)
        if d.found:
            dets.append(d)
    if len(dets) < 8:
        return None
    r = calibrate(dets, rig.image_size)
    return {
        "label": label,
        "z_min": float(zs.min()), "z_max": float(zs.max()),
        "span": float(zs.max() / zs.min()),
        "n_used": len(dets), "n_total": len(imgs),
        "fx": r.fx, "fy": r.fy, "cx": r.cx, "cy": r.cy,
        "k1": float(r.dist[0]), "k2": float(r.dist[1]), "k3": float(r.dist[4]),
        "fx_err_pct": (r.fx - TRUE["fx"]) / TRUE["fx"] * 100,
        "rms": r.rms,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="深度多样性 vs 内参可辨识性（受控实验）")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--views", type=int, default=0, help=">0 时按它折算 rate")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rate = args.rate
    if args.views > 0:
        rate = args.views / args.duration

    spec = BoardSpec("chessboard", (9, 6), 0.025)
    print("=" * 78)
    print("受控实验：只改**深度多样性**，真值 K 完全不变")
    print("=" * 78)
    print(f"真值  fx={TRUE['fx']}  fy={TRUE['fy']}  cx={TRUE['cx']}  cy={TRUE['cy']}"
          f"   k1={TRUE['k1']}  k2={TRUE['k2']}  k3={TRUE['k3']}\n")

    rows = []
    for label, pa, cz in CASES:
        r = run_case(label, pa, cz, spec, args.duration, rate, args.seed)
        if r is None:
            print(f"  {label}: 检出太少，跳过")
            continue
        rows.append(r)
        print(f"  {label:22s} 深度 {r['z_min']:.2f}~{r['z_max']:.2f} m"
              f"（跨度 {r['span']:.1f}x）")
        print(f"     解出 fx = {r['fx']:8.3f}   误差 {r['fx_err_pct']:+7.2f}%   "
              f"k1={r['k1']:+.5f} k2={r['k2']:+.5f} k3={r['k3']:+.5f}   "
              f"rms={r['rms']:.4f} px")
        print()

    print("=" * 78)
    print("结论")
    print("=" * 78)
    if rows:
        worst = max(rows, key=lambda r: abs(r["fx_err_pct"]))
        best = min(rows, key=lambda r: abs(r["fx_err_pct"]))
        rms_spread = max(r["rms"] for r in rows) - min(r["rms"] for r in rows)
        print(f"  fx 误差：最差 {worst['fx_err_pct']:+.2f}%（{worst['label']}）"
              f"  →  最好 {best['fx_err_pct']:+.2f}%（{best['label']}）")
        print(f"  但**重投影误差只差 {rms_spread:.4f} px** —— "
              f"光看残差分辨不出内参对错。")
        print()
        print("  ⇒ 采集时必须让标定板走遍**近/中/远三档距离**，"
              "深度跨度至少 2x。")
        print("     只在同一距离换角度 → fx 与畸变简并，内参不可信，"
              "而且残差不会报警。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
