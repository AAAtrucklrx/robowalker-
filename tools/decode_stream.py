#!/usr/bin/env python3
"""把 IMU 板的二进制流解码成 CSV —— 帧结构已从原始字节反推出来。

帧结构（从 8045 帧实测反推，自相关一致率 100%）：

    ┌─────────────── 28 字节 ───────────────┐
    │ 头 4 字节 │ 数据 24 字节（6 × float32） │
    │ ab cd 00 00│ 陀螺 x/y/z + 加速度 x/y/z   │
    └───────────┴────────────────────────────┘

依据：
  · 28 字节自相关完美重复，帧率 ~1 kHz（28,000 B/s ÷ 28）
  · 28 = 4 + 24，而 24 = 6 × float32 —— IMU 六轴最标准的排法
  · 头里 `00 00 71 3d` 按小端 float32 = 0.0588，像是采样间隔/数据率参数

⚠️ 当前板子的数据区全是 0（传感器未被采样），所以解码出来是 0。
   本脚本的作用是：**等板子修好后立刻能用**，并顺手验证"头 4 + 数据 24"这个假设。

用法:
    .venv/bin/python tools/decode_stream.py --port /dev/rw_imu --seconds 5
    .venv/bin/python tools/decode_stream.py --file /tmp/imu_raw.bin --csv out.csv
    .venv/bin/python tools/decode_stream.py --file x.bin --header 4 --layout gyro_accel
    .venv/bin/python tools/decode_stream.py --file x.bin --autodetect   # 自动试多种布局
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

FRAME = 28

# 可尝试的帧布局：名字 -> (头长度, 数据字段名列表)
LAYOUTS = {
    "gyro_accel": (4, ["gx", "gy", "gz", "ax", "ay", "az"]),
    "accel_gyro": (4, ["ax", "ay", "az", "gx", "gy", "gz"]),
    "gyro_accel_ts": (8, ["gx", "gy", "gz", "ax", "ay", "az", "t"]),
    "int16_gyro_accel": (4, ["gx", "gy", "gz", "ax", "ay", "az"]),  # 用 int16，见 dtype
}


def split_frames(data: bytes, frame: int = FRAME) -> np.ndarray:
    n = len(data) // frame
    if n == 0:
        raise SystemExit(f"数据不足一帧（{len(data)} 字节 < {frame}）")
    a = np.frombuffer(data[: n * frame], dtype=np.uint8).reshape(n, frame)
    return a


def check_header(frames: np.ndarray, hdr_len: int) -> bool:
    """所有帧的头字节是否完全一致（定长协议的标志）。"""
    return bool((frames[:, :hdr_len] == frames[0, :hdr_len]).all())


def decode(frames: np.ndarray, hdr_len: int, dtype: str, endian: str = "<") -> np.ndarray:
    body = frames[:, hdr_len:]
    n = len(frames)
    if dtype == "float32":
        vals = body.reshape(n, -1, 4).copy().view(np.dtype(endian + "f4")).reshape(n, -1)
    elif dtype == "int16":
        vals = body.reshape(n, -1, 2).copy().view(np.dtype(endian + "i2")).reshape(n, -1)
    else:
        raise ValueError(dtype)
    return vals.astype(np.float64)


def report(vals: np.ndarray, names: list[str], label: str) -> bool:
    """打印每列的统计。全 0 的列说明该通道没有数据 —— 这就是"数据区是死的"的判据。"""
    print(f"\n=== 解码结果 {label} ===")
    print(f"{'字段':<6}{'均值':>14}{'标准差':>14}{'最小值':>14}{'最大值':>14}  状态")
    live = 0
    for i, name in enumerate(names):
        if i >= vals.shape[1]:
            break
        c = vals[:, i]
        std = float(c.std())
        state = "✅ 有数据" if std > 1e-9 else "❌ 恒为 0"
        live += std > 1e-9
        print(f"{name:<6}{c.mean():>14.6f}{std:>14.6f}{c.min():>14.6f}{c.max():>14.6f}  {state}")
    print(f"\n有数据的通道: {live} / {vals.shape[1]}")
    if live == 0:
        print("→ 所有通道恒定。板子仍在故障态（传感器未被采样），不是解码问题。")
        return False
    print("→ 有通道在变化，帧结构假设成立，可以进入标定流程。")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=Path)
    src.add_argument("--port")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--frame", type=int, default=FRAME)
    ap.add_argument("--header", type=int, default=4, help="帧头字节数")
    ap.add_argument("--dtype", default="float32", choices=["float32", "int16"])
    ap.add_argument("--endian", default="<", choices=["<", ">"])
    ap.add_argument("--layout", default="gyro_accel", choices=sorted(LAYOUTS))
    ap.add_argument("--csv", type=Path, help="把解码结果写成 CSV")
    ap.add_argument("--autodetect", action="store_true", help="自动尝试多种布局")
    args = ap.parse_args()

    if args.file:
        data = args.file.read_bytes()
        label = str(args.file)
    else:
        import serial
        ser = serial.Serial(args.port, 115200, timeout=0.2)
        import time
        t0 = time.monotonic()
        buf = []
        while time.monotonic() - t0 < args.seconds:
            c = ser.read(65536)
            if c:
                buf.append(c)
        ser.close()
        data = b"".join(buf)
        label = args.port

    print(f"数据源: {label}   {len(data)} 字节")
    frames = split_frames(data, args.frame)
    print(f"按 {args.frame} 字节切帧: {len(frames)} 帧")

    if args.autodetect:
        print("\n=== 自动尝试各种布局 ===")
        best = None
        for name, (hdr, names) in LAYOUTS.items():
            if not check_header(frames, hdr):
                print(f"  {name:<20} 头 {hdr} 字节不一致 → 跳过")
                continue
            dt = "int16" if "int16" in name else "float32"
            try:
                vals = decode(frames, hdr, dt, args.endian)
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:<20} 解码失败: {exc}")
                continue
            live = int((vals.std(axis=0) > 1e-9).sum())
            sane = bool(np.isfinite(vals).all())
            print(f"  {name:<20} 头一致 ✅  通道 {vals.shape[1]}  有数据 {live}  "
                  f"数值正常 {sane}  量级 {np.abs(vals).max():.3g}")
            if sane and (best is None or live > best[1]):
                best = (name, live, vals, names)
        if best is None:
            print("\n没有可用的布局")
            return 2
        name, live, vals, names = best
        print(f"\n最佳布局: {name}")
        report(vals, names, f"{label} [{name}]")
        return 0 if live else 1

    hdr, names = LAYOUTS[args.layout]
    if args.layout == "int16_gyro_accel":
        args.dtype = "int16"
    if not check_header(frames, hdr):
        print(f"⚠️  头 {hdr} 字节在各帧之间不一致，帧结构假设可能有误")
    vals = decode(frames, hdr, args.dtype, args.endian)
    ok = report(vals, names, f"{label} [header={hdr}, {args.dtype}]")

    if args.csv:
        import csv
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_index"] + names[: vals.shape[1]])
            for i, row in enumerate(vals):
                w.writerow([i] + [f"{v:.9g}" for v in row])
        print(f"\n已写出 {args.csv}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
