#!/usr/bin/env python3
"""串口原始字节抓取器 —— 在写任何解析代码之前，先看清设备到底吐什么。

和 capture_dataset.py 的区别：这个工具**不做任何解析**，只统计字节、行、
ASCII 可见性与疑似二进制帧头，因此设备协议再古怪也能看清。

用法:
    # 抓 10 秒，打印可读文本行 + 统计
    .venv/bin/python tools/serial_probe.py --port /dev/ttyACM0 --seconds 10

    # 存原始字节，便于反复分析
    .venv/bin/python tools/serial_probe.py --port /dev/ttyACM0 --dump raw.bin

    # 波特率对 USB CDC 无效（ttyACM*），但对 ttyUSB* 有效
    .venv/bin/python tools/serial_probe.py --port /dev/ttyUSB0 --baud 921600
"""
from __future__ import annotations

import argparse
import collections
import sys
import time

import serial


def visible(b: int) -> str:
    if 32 <= b < 127:
        return chr(b)
    if b == 0x0A:
        return "\\n"
    if b == 0x0D:
        return "\\r"
    if b == 0x09:
        return "\\t"
    return f"<{b:02X}>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--max-lines", type=int, default=40, help="最多打印多少行样本")
    ap.add_argument("--dump", default=None, help="把原始字节写到文件")
    args = ap.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.2)
    except PermissionError:
        print(f"❌ 打开 {args.port} 权限不足。")
        print("   先跑: sudo bash ~/calib_ws/tools/setup_permissions.sh")
        print("   或临时: sudo chmod a+rw " + args.port)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 打开 {args.port} 失败: {exc}")
        return 1

    print(f"=== 抓取 {args.port} @ {args.baud} ，持续 {args.seconds:.0f} 秒 ===")
    if args.port.startswith("/dev/ttyACM"):
        print("（注意：USB CDC 虚拟串口的波特率由设备固件决定，这里的值不影响通信）")

    t0 = time.monotonic()
    buf = bytearray()
    lines: list[str] = []
    byte_hist: collections.Counter[int] = collections.Counter()
    n_bytes = 0

    try:
        while time.monotonic() - t0 < args.seconds:
            chunk = ser.read(4096)
            if not chunk:
                continue
            buf.extend(chunk)
            byte_hist.update(chunk)
            n_bytes += len(chunk)
            while b"\n" in buf:
                raw, _, rest = buf.partition(b"\n")
                buf = bytearray(rest)
                if len(lines) < args.max_lines:
                    txt = raw.decode("utf-8", errors="replace").rstrip("\r")
                    lines.append(txt)
    except KeyboardInterrupt:
        print("\n（手动中断）")
    finally:
        elapsed = time.monotonic() - t0
        if args.dump:
            with open(args.dump, "wb") as f:
                f.write(bytes(buf))
            print(f"尾部残余 {len(buf)} 字节已写入 {args.dump}")
        ser.close()

    print(f"\n收到 {n_bytes} 字节 / {elapsed:.1f}s  →  {n_bytes / max(elapsed, 1e-6):.0f} B/s")

    if n_bytes == 0:
        print("\n❌ 一个字节都没收到。可能原因（按概率排序）：")
        print("   1. 固件因为自检失败停在了错误状态，根本没进数据发送循环 ← 与你听到的滴滴声一致")
        print("   2. 设备需要主机先发一条启动/使能指令才会开始发数据")
        print("   3. 线是充电线而非数据线（但已经枚举出 CDC，基本可排除）")
        print("   4. 打开的设备节点不对（试试 /dev/ttyACM1 或 /dev/serial/by-id/ 里的名字）")
        return 2

    printable = sum(v for k, v in byte_hist.items() if 32 <= k < 127 or k in (10, 13, 9))
    ratio = printable / max(n_bytes, 1)
    print(f"可打印字符占比 {ratio * 100:.1f}%  →  {'ASCII 文本协议' if ratio > 0.9 else '疑似二进制协议'}")
    print(f"换行符 \\n 出现 {byte_hist[10]} 次  →  约 {byte_hist[10] / max(elapsed, 1e-6):.1f} 行/秒（若一行一帧即数据率）")

    print(f"\n--- 前 {len(lines)} 行样本 ---")
    for i, ln in enumerate(lines):
        print(f"[{i:3d}] {ln[:160]}")

    print("\n--- 最常见的 16 个字节（十六进制）---")
    for byte, cnt in byte_hist.most_common(16):
        print(f"  0x{byte:02X} {cnt:>8d}  ({visible(byte)})")

    heads = collections.Counter()
    for i in range(max(0, n_bytes - 3)):
        pass  # 逐字节扫描成本高，这里只在样本上做
    print("\n--- 协议初判提示 ---")
    if ratio > 0.9 and byte_hist[10] > 10:
        print("  → 文本行协议：把上面任一行发我，我直接写正则解析")
    elif byte_hist[0x55] > 10:
        print("  → 大量 0x55 帧头：非常像维特 WIT 二进制协议（0x55 0x51 加速度 / 0x52 角速度 / 0x53 欧拉角）")
    elif byte_hist[0xA5] > 10:
        print("  → 大量 0xA5 帧头：像维特/HiWonder 的另一种帧头约定")
    else:
        print("  → 帧头不明确，请把 --dump 出来的 raw.bin 交给我做逐字节分析")
    return 0


if __name__ == "__main__":
    sys.exit(main())
