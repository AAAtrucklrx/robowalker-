#!/usr/bin/env python3
"""IMU/相机数据的运动激励体检 —— 判断这组数据够不够做 Camera-IMU 标定。

为什么需要它：手眼标定 (AX = XB) 解不出唯一解，常见原因不是算法错，而是
采集时运动太单调（只绕一个轴转、没有平移激励、或者全程静止）。
在动手写标定之前先体检，能省掉几个小时的瞎调参。

用法:
    .venv/bin/python tools/check_motion.py --imu data/session_01/imu.txt
    .venv/bin/python tools/check_motion.py --imu ... --show        # 顺手画曲线
    .venv/bin/python tools/check_motion.py --imu ... --unit ms     # 时间戳单位
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load_imu(path: Path, unit: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[unit]
    rows = []
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.replace(",", " ").split()
        if len(parts) < 7:
            continue
        try:
            rows.append([float(x) for x in parts[:7]])
        except ValueError:
            continue
    if not rows:
        raise SystemExit(f"{path} 里没解析出任何样本（每行需要 7 个数：t gx gy gz ax ay az）")
    a = np.asarray(rows, dtype=np.float64)
    t = a[:, 0] * scale
    return t, a[:, 1:4], a[:, 4:7]


def fmt(v: float, unit: str = "") -> str:
    return f"{v:8.4f}{unit}"


def analyze(t: np.ndarray, gyro: np.ndarray, accel: np.ndarray) -> dict:
    dt = np.diff(t)
    dt = dt[dt > 0]
    fs = 1.0 / np.median(dt) if dt.size else float("nan")
    dur = float(t[-1] - t[0])
    gap_max = float(np.max(dt)) if dt.size else 0.0

    # 静止检测：滑窗标准差，用于零偏估计与初始姿态对齐
    #
    # 注意：必须算**窗内去均值**的平方和。早先版本用 (a - 全局均值)² 求窗内均值，
    # 会把静止段那 9.8 m/s² 的重力整体算成"波动"，导致静止段永远被判成 0%。
    win = max(int(fs * 0.2), 5)
    if len(t) > win * 2:
        k = np.ones(win) / win

        def win_std(x: np.ndarray) -> np.ndarray:
            """逐窗标准差（沿时间轴）：sqrt(E[x²] − E[x]²)。"""
            m = np.convolve(x, k, "valid")
            m2 = np.convolve(x * x, k, "valid")
            return np.sqrt(np.maximum(m2 - m * m, 0.0))

        gs = win_std(np.linalg.norm(gyro, axis=1))
        asd = win_std(np.linalg.norm(accel, axis=1))
        static = (gs < 0.05) & (asd < 0.15)
        static_ratio = float(static.mean())
        # 最长连续静止段：标定需要一段完整静止来做零偏，零散的静止点没用
        longest = 0
        cur = 0
        for flag in static:
            cur = cur + 1 if flag else 0
            longest = max(longest, cur)
        longest_static = longest / fs
    else:
        static_ratio = 0.0
        longest_static = 0.0

    # 旋转轴多样性：把每个时刻的角速度方向归一化，看主轴覆盖了多少方向
    gnorm = np.linalg.norm(gyro, axis=1)
    moving = gnorm > np.percentile(gnorm, 60)
    axes = gyro[moving] / np.maximum(gnorm[moving, None], 1e-9)
    if len(axes) > 10:
        cov = axes.T @ axes / len(axes)
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        # 三个特征值越接近 1/3，说明旋转轴越均匀分布（激励越充分）
        axis_spread = float(eig[-1] / max(eig[0], 1e-9))
    else:
        eig = np.zeros(3)
        axis_spread = 0.0

    # 加速度模长：静止时应接近重力；变化范围反映平动激励
    anorm = np.linalg.norm(accel, axis=1)
    g_est = float(np.median(anorm))

    # 频域：看能量是否集中在极低频（说明只是缓慢挪动，缺少动态激励）
    if len(gyro) > 64:
        spec = np.abs(np.fft.rfft(gyro - gyro.mean(0), axis=0)).sum(1)
        freqs = np.fft.rfftfreq(len(gyro), 1.0 / fs)
        band = freqs > max(fs * 0.005, 0.05)
        low_ratio = float(spec[~band].sum() / max(spec.sum(), 1e-12))
    else:
        low_ratio = 1.0

    return {
        "n": len(t), "fs": fs, "dur": dur, "gap_max": gap_max,
        "static_ratio": static_ratio, "longest_static": longest_static,
        "gyro_max": float(np.abs(gyro).max()),
        "gyro_rms": float(np.sqrt((gyro ** 2).sum(1).mean())),
        "accel_range": float(anorm.max() - anorm.min()),
        "accel_max_dev": float(np.abs(anorm - g_est).max()),
        "g_est": g_est,
        "axis_eig": eig, "axis_spread": axis_spread,
        "low_freq_ratio": low_ratio,
        "gyro_bias": gyro.mean(0) if static_ratio > 0.2 else np.zeros(3),
    }


def verdicts(m: dict) -> list[tuple[bool, str]]:
    out: list[tuple[bool, str]] = []
    out.append((100 <= m["fs"] <= 2000,
                f"采样率 {m['fs']:.0f} Hz —— 标定一般要 100 Hz 以上；过低会导致时间对齐误差被放大"))
    out.append((m["gap_max"] < 5 * max(1.0 / max(m["fs"], 1), 1e-3) * 10,
                f"最大采样间隔 {m['gap_max'] * 1e3:.1f} ms —— 明显丢包会造成积分漂移"))
    out.append((m["dur"] >= 20.0,
                f"有效时长 {m['dur']:.1f} s —— 建议 ≥ 20 s，且包含丰富的转动与小幅平移"))
    out.append((m["longest_static"] >= 1.0,
                f"最长连续静止段 {m['longest_static']:.2f} s（占比 {m['static_ratio'] * 100:.0f}%）"
                f" —— 开头需要连续 1-2 s 完全静止，用于零偏标定与重力对齐；零散的静止点不算数"))
    out.append((m["gyro_max"] >= 0.5,
                f"角速度峰值 {m['gyro_max']:.2f} rad/s ({np.degrees(m['gyro_max']):.0f} deg/s) —— 太温和则旋转外参的可观测性差"))
    out.append((m["axis_spread"] >= 0.4,
                f"旋转轴多样性 {m['axis_spread']:.2f} (理想接近 1.0) —— 只绕一个轴转，是手眼标定无解/病态的头号原因"))
    out.append((m["accel_max_dev"] >= 0.5,
                f"加速度偏离重力最大 {m['accel_max_dev']:.2f} m/s² —— 平移激励不足会让平移外参不可观"))
    out.append((m["low_freq_ratio"] <= 0.9,
                f"低频能量占比 {m['low_freq_ratio'] * 100:.0f}% —— 太高说明动作过慢，缺少动态激励"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imu", required=True, type=Path)
    ap.add_argument("--unit", default="s", choices=["s", "ms", "us", "ns"])
    ap.add_argument("--show", action="store_true", help="画图查看（需要图形界面）")
    args = ap.parse_args()

    t, gyro, accel = load_imu(args.imu, args.unit)
    m = analyze(t, gyro, accel)

    print(f"=== IMU 数据体检: {args.imu} ===")
    print(f"样本数        {m['n']}")
    print(f"时长          {m['dur']:.2f} s")
    print(f"采样率        {m['fs']:.1f} Hz")
    print(f"重力估计      {m['g_est']:.3f} m/s²   (预期 ≈ 9.78-9.82)")
    print(f"角速度 RMS    {m['gyro_rms']:.3f} rad/s   峰值 {m['gyro_max']:.3f} rad/s")
    print(f"加速度范围    {m['accel_range']:.3f} m/s²")
    print(f"静止段占比    {m['static_ratio'] * 100:.1f}%")
    print(f"旋转轴特征值  {np.round(m['axis_eig'], 4)}   多样性 {m['axis_spread']:.2f}")

    print("\n--- 逐项判定 ---")
    passed = 0
    for ok, text in verdicts(m):
        print(f"{'✅' if ok else '❌'} {text}")
        passed += ok
    total = len(verdicts(m))
    print(f"\n通过 {passed}/{total} 项")
    if passed < total:
        print("建议: 重新采集。流程 = 静止 2s → 缓慢绕 x/y/z 各转几圈 → 手持小幅平移 → 画 8 字 → 全程避免急停")
    else:
        print("数据激励合格，可以进入标定阶段。")

    if args.show:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
        ax[0].plot(t - t[0], gyro)
        ax[0].set_ylabel("gyro [rad/s]"); ax[0].legend(["gx", "gy", "gz"])
        ax[1].plot(t - t[0], accel)
        ax[1].set_ylabel("accel [m/s²]"); ax[1].legend(["ax", "ay", "az"])
        ax[1].set_xlabel("time [s]")
        plt.tight_layout(); plt.show()
    return 0 if passed == total else 2


if __name__ == "__main__":
    sys.exit(main())
