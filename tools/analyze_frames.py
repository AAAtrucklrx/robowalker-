#!/usr/bin/env python3
"""串口帧结构诊断 —— 自动判定定长二进制协议，并检查数据区是否真的在更新。

动机：STMicroelectronics CDC 的 IMU 板可能「帧头在发、数据区恒为 0」——
看起来有 1 kHz 的数据流，实际传感器根本没被采样。肉眼和字节统计都看不出来，
必须做帧切割 + 数据区熵检查。

用法:
    .venv/bin/python tools/analyze_frames.py --file /tmp/imu_raw_rw_imu.bin
    .venv/bin/python tools/analyze_frames.py --port /dev/rw_imu --seconds 5
    .venv/bin/python tools/analyze_frames.py --file x.bin --frame 28     # 指定帧长
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np


def autoregressive_period(data: bytes, max_period: int = 256) -> tuple[int, float]:
    """用自相关找最短重复周期：对每个候选周期 p，统计 data[i] == data[i+p] 的比例。"""
    a = np.frombuffer(data[: min(len(data), 200_000)], dtype=np.uint8)
    best_p, best_score = 0, 0.0
    for p in range(2, max_period):
        n = len(a) - p
        if n < 1000:
            break
        score = float((a[:n] == a[p : p + n]).mean())
        if score > best_score:
            best_p, best_score = p, score
    return best_p, best_score


def analyze(data: bytes, frame_len: int | None = None, label: str = "") -> int:
    print(f"=== 帧结构诊断 {label} ===")
    print(f"总字节 {len(data)}")
    if len(data) < 1000:
        print("样本太少")
        return 2

    if frame_len is None:
        p, score = autoregressive_period(data)
        print(f"自相关探测到最短重复周期 = {p} 字节（一致率 {score * 100:.2f}%）")
        frame_len = p
        if score < 0.99:
            print("⚠️  一致率不足 99%，可能不是严格定长帧，或存在丢字节")
    if frame_len < 4:
        print("周期过短，判定为无效/常值流")
        return 1

    n_frames = len(data) // frame_len
    tail = len(data) % frame_len
    print(f"按 {frame_len} 字节切帧: {n_frames} 帧，尾部余 {tail} 字节")
    print(f"帧率估算（按 15 秒实测吞吐 28000 B/s 换算）≈ {28000 / frame_len:.0f} Hz")

    frames = np.frombuffer(data[: n_frames * frame_len], dtype=np.uint8)
    frames = frames.reshape(n_frames, frame_len)

    # 1. 帧头常量性：找出所有帧都相同的字节位置
    const_mask = (frames == frames[0]).all(axis=0)
    const_pos = np.flatnonzero(const_mask)
    print(f"\n帧内恒定字节位置: {const_pos.tolist()}")
    print(f"  帧头常量 = {frames[0][const_pos].tobytes().hex(' ')}")

    # 2. 每个字节位置的取值个数（熵）
    uniq = np.array([len(np.unique(frames[:, j])) for j in range(frame_len)])
    var_pos = np.flatnonzero(uniq > 1)
    print(f"\n有变化的字节位置数: {len(var_pos)} / {frame_len}")
    for j in range(frame_len):
        tag = "恒定" if uniq[j] == 1 else f"{uniq[j]} 种取值"
        print(f"  字节[{j:2d}] {tag:<12} 样本值: "
              f"{' '.join(f'{v:02x}' for v in np.unique(frames[:, j])[:6])}")

    # 3. 数据区是否在更新 —— 这是判断"传感器有没有被采样"的核心
    if len(var_pos) == 0:
        print("\n❌ 全部帧完全相同：设备在重复发送同一份（常量）帧。")
        print("   → 传感器没有被采样，或固件卡在初始化错误循环。")
        return 1

    changing_ratio = len(var_pos) / frame_len
    print(f"\n变化字节占比 {changing_ratio * 100:.1f}%")

    # 4. 把"有变化的字节"解码成几种可能的数值类型，看哪种更合理
    print("\n--- 尝试解码变化区域 ---")
    for name, dtype, size in (("int16", np.int16, 2), ("uint16", np.uint16, 2),
                              ("float32", np.float32, 4), ("uint32", np.uint32, 4)):
        # 只在有变化的字节区间按对齐方式切
        start = int(var_pos[0]) if len(var_pos) else 0
        start -= start % size
        usable = ((frame_len - start) // size) * size
        if usable < size:
            continue
        region = frames[:, start : start + usable].reshape(n_frames, -1, size)
        for endian in ("<", ">"):
            try:
                vals = region.reshape(n_frames, -1).copy().view(dtype.newbyteorder(endian))
            except Exception:  # noqa: BLE001
                continue
            if not np.isfinite(vals.astype(np.float64)).all():
                continue
            spread = np.abs(vals.astype(np.float64)).max()
            if spread == 0:
                continue
            print(f"  以 {endian}{name} 解码（自偏移 {start} 起 {usable} 字节 → "
                  f"{usable // size} 个值/帧）：")
            for k in range(min(usable // size, 6)):
                col = vals[:, k].astype(np.float64)
                print(f"    值[{k}] 均值 {col.mean():12.4f}  标准差 {col.std():10.4f}  "
                      f"范围 [{col.min():.4f}, {col.max():.4f}]")
            break
        break

    print("\n--- 结论 ---")
    if changing_ratio > 0.2:
        print("✅ 数据区在持续变化：帧头+数据帧结构成立，传感器有输出。")
        print("   下一步：确认哪个字节对应哪个物理量（动一下板子看哪几列跟着变）。")
        return 0
    print("⚠️  只有少数字节在变化，数据区大部分恒定。")
    print("   可能是：序列号/计数器在变，但传感器数据区仍是死的。")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path)
    ap.add_argument("--port")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--frame", type=int, default=None, help="手动指定帧长")
    args = ap.parse_args()

    if args.file:
        data = args.file.read_bytes()
        return analyze(data, args.frame, label=str(args.file))

    if args.port:
        import serial
        ser = serial.Serial(args.port, 115200, timeout=0.2)
        chunks = []
        import time
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.seconds:
            c = ser.read(65536)
            if c:
                chunks.append(c)
        ser.close()
        data = b"".join(chunks)
        print(f"从 {args.port} 抓取 {len(data)} 字节 / {args.seconds:.0f}s\n")
        return analyze(data, args.frame, label=args.port)

    ap.error("需要 --file 或 --port")
    return 2


if __name__ == "__main__":
    sys.exit(main())
