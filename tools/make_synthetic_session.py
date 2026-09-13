#!/usr/bin/env python3
"""生成一份**落盘的**合成数据集，用于端到端验证 `run_calibration.py`。

与 ``tools/selftest_pipeline.py`` 的区别
------------------------------------------------------------------
那个是在内存里跑；这个**把图像、时间戳、IMU 全部写到磁盘**，目录结构
与真实采集完全一致，然后让 ``calib/run_calibration.py`` 从文件读。
只有这样才能验证交付物本身（路径解析、格式解析、配置驱动）是对的。

刻意模拟了两个真实世界的"麻烦"：

1. **IMU 时间戳用板载时钟**（不是主机时钟），与主机时钟有一个未知常量偏移
   —— 这正是 C2 存在的理由。真机上 H7 板给的就是毫秒计数器；
2. 图像分辨率、标定板参数、噪声都从命令行给，不写死在代码里。

产出::

    <out>/
    ├── images/000000.png ...
    ├── image_timestamps.txt      主机单调时钟（秒）
    ├── imu.txt                   **板载时钟**（秒）+ 六轴，t_host = t_imu + tau
    ├── truth.json                真值 T_IC / tau / 内参，仅用于验证，标定程序不读
    └── README.md                 说明这份数据是怎么来的

用法::

    .venv/bin/python tools/make_synthetic_session.py --out data/synth_01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calib"))

from simulate import SimRig, Trajectory, describe  # noqa: E402
from target_detect import BoardSpec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="生成落盘的合成 Camera-IMU 数据集")
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=9.0)
    ap.add_argument("--cam-rate", type=float, default=20.0)
    ap.add_argument("--imu-rate", type=float, default=1000.0)
    ap.add_argument("--pattern", default="9x6")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--size", default="1280x960")
    ap.add_argument("--supersample", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lead-in", type=float, default=2.5,
                    help="开头静止段时长（真实采集协议要求 ≥2 s）")
    ap.add_argument("--tau", type=float, default=0.137,
                    help="IMU 板载时钟相对主机时钟的偏移（秒）")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out).expanduser()
    (out / "images").mkdir(parents=True, exist_ok=True)
    for old in (out / "images").glob("*.png"):
        old.unlink()

    W, H = (int(v) for v in args.size.lower().split("x"))
    c, r = (int(v) for v in args.pattern.lower().split("x"))
    spec = BoardSpec("chessboard", (c, r), args.square_mm / 1000.0)

    # ⚠️ **深度多样性是内参可辨识的硬条件**，不是可选项。
    # 受控实验（同一真值 fx=1100，只改深度跨度）：
    #     跨度 1.0x（相机几乎不动）-> fx 偏 +4.17%，k3 跑到 -0.047
    #     跨度 1.8x                -> fx 偏 -0.12%
    #     跨度 3.3x                -> fx 偏 +0.31%，k1/k2/k3 全部接近真值
    # 三种情况的**重投影误差都是 0.13~0.15 px** —— 光看残差发现不了简并。
    # 所以这里刻意让 z 在 0.70~1.80 m 之间摆动（跨度 ~2.6x）。
    rig = SimRig(traj=Trajectory(rot_amp=np.array([0.20, 0.15, 0.26]),
                                 rot_freq=np.array([2.6, 3.6, 1.5]),
                                 pos_amp=np.array([0.05, 0.04, 0.55]),
                                 pos_freq=np.array([0.8, 1.2, 0.45]),
                                 center=np.array([0.0, 0.0, 1.25]),
                                 lead_in_s=args.lead_in),
                 tau=args.tau)
    rig.K = np.array([[1100.0, 0, W / 2 - 12.0],
                      [0, 1098.0, H / 2 + 8.0], [0, 0, 1.0]])
    rig.image_size = (W, H)

    print(describe(rig))
    print(f"\n标定板 {spec}")
    print(f"渲染 {W}x{H}，{args.duration}s @ {args.cam_rate}Hz "
          f"（超采样 {args.supersample}x）...")

    # ── 图像 + 主机时间戳 ───────────────────────────────────
    cam_times, imgs, rvec_true, tvec_true = rig.make_camera_stream(
        0.0, args.duration, rate=args.cam_rate, board_spec=spec,
        supersample=args.supersample, noise_sigma=0.0, rng=rng)
    lines = ["# filename timestamp_seconds (host monotonic clock, unit: s)"]
    for k, img in enumerate(imgs):
        name = f"{k:06d}.png"
        cv2.imwrite(str(out / "images" / name), img)
        lines.append(f"{name} {cam_times[k]:.9f}")
    (out / "image_timestamps.txt").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")
    print(f"  图像 {len(imgs)} 张")

    # ── IMU：**板载时钟** ───────────────────────────────────
    pad = 0.5
    _, t_imu, gyro, accel = rig.make_imu_stream(
        -pad, args.duration + pad, rate=args.imu_rate, rng=rng, with_noise=True)
    with (out / "imu.txt").open("w", encoding="utf-8") as f:
        f.write("# timestamp gx gy gz ax ay ax   "
                "(s, rad/s x3, m/s^2 x3; timestamp = IMU 板载时钟)\n")
        for i in range(len(t_imu)):
            g = gyro[i]; a = accel[i]
            f.write(f"{t_imu[i]:.9f} {g[0]:.6f} {g[1]:.6f} {g[2]:.6f} "
                    f"{a[0]:.6f} {a[1]:.6f} {a[2]:.6f}\n")
    print(f"  IMU {len(t_imu)} 条（板载时钟，偏移 {args.tau*1000:+.1f} ms）")

    # ── 真值（标定程序不读，仅供验证）────────────────────────
    truth = {
        "note": "标定程序不会读这个文件；它只用于验证标定结果",
        "T_IC_camera_to_imu": rig.T_IC().tolist(),
        "R_IC": rig.R_IC.tolist(),
        "t_IC": rig.t_IC.tolist(),
        "lever_arm_in_camera_m": rig.r_lever.tolist(),
        "tau_s": args.tau,
        "tau_convention": "t_host = t_imu + tau",
        "K": rig.K.tolist(),
        "dist": rig.dist.tolist(),
        "image_size": [W, H],
        "board": {"type": "chessboard", "pattern_size": [c, r],
                  "square_size_m": spec.square_size},
        "n_images": len(imgs),
        "n_imu": len(t_imu),
        "cam_rate_hz": args.cam_rate,
        "imu_rate_hz": args.imu_rate,
    }
    (out / "truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")

    (out / "README.md").write_text(f"""# 合成数据集（{out.name}）

由 `tools/make_synthetic_session.py` 生成，用于**端到端验证** `calib/run_calibration.py`。

{describe(rig)}

- 标定板：{spec}
- 分辨率：{W}x{H}，相机 {args.cam_rate} Hz，IMU {args.imu_rate} Hz，{args.duration} s
- **IMU 时间戳用的是板载时钟**，与主机时钟相差 {args.tau*1000:+.1f} ms
  （`t_host = t_imu + tau`）——这正是 C2 要估的量。
- `truth.json` 是真值，**标定程序不会读它**，只有验证脚本会用。

跑法：

```bash
.venv/bin/python calib/run_calibration.py --session {out} \\
    --pattern {c}x{r} --square-mm {args.square_mm}
```
""", encoding="utf-8")

    print(f"\n✅ 已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
