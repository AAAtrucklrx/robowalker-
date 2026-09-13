#!/usr/bin/env python3
"""离线自测：用一个伪串口伪造 IMU 数据流，验证「解析 → 落盘 → 体检」整条链路。

在没有真硬件、或硬件还在报错时，用它证明解析与检查逻辑是对的，
这样等真 IMU 一接上就只剩"协议是否匹配"这一个变量。

用法:
    .venv/bin/python tools/selftest_serial.py                 # 用内置的维特 ASCII 格式
    .venv/bin/python tools/selftest_serial.py --format csv    # 换成数字列格式
    .venv/bin/python tools/selftest_serial.py --algo-deg      # 模拟"单位是 deg/s、g"的设备
"""
from __future__ import annotations

import argparse
import math
import os
import pty
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def make_stream(fmt: str, algo_deg: bool, fs: float, dur: float, stop: threading.Event):
    """生成一段"手握 IMU 缓慢转动 + 静止"的合成数据。"""
    n = int(fs * dur)
    dt = 1.0 / fs
    for i in range(n):
        if stop.is_set():
            return
        t = i * dt
        # 前 2 秒静止（用于零偏/重力对齐），之后绕三轴做不同频率的转动
        if t < 2.0:
            gx = gy = gz = 0.02 * math.sin(t * 3)
            ax, ay, az = 0.01, -0.02, 9.80
        else:
            gx = 1.2 * math.sin(2 * math.pi * 0.3 * t)
            gy = 0.9 * math.sin(2 * math.pi * 0.21 * t + 1.0)
            gz = 0.7 * math.sin(2 * math.pi * 0.13 * t + 2.0)
            ax = 9.80 + 1.5 * math.sin(2 * math.pi * 0.45 * t)
            ay = 0.8 * math.sin(2 * math.pi * 0.37 * t)
            az = -0.5 * math.sin(2 * math.pi * 0.29 * t)
        ts = 1000.0 + t
        if fmt == "wit":
            # 维特 ASCII 风格: t=... gx=... ay=...  设备单位可能是 deg/s 和 g
            g = 180.0 / math.pi if algo_deg else 1.0
            a = 1.0 / 9.81 if algo_deg else 1.0
            line = (f"t={ts:.3f} gx={gx * g:.4f} gy={gy * g:.4f} gz={gz * g:.4f} "
                    f"ax={ax * a:.4f} ay={ay * a:.4f} az={az * a:.4f}\r\n")
        else:
            line = (f"{ts:.4f},{gx:.6f},{gy:.6f},{gz:.6f},"
                    f"{ax:.6f},{ay:.6f},{az:.6f}\n")
        os.write(MASTER_FD, line.encode())
        time.sleep(dt)


MASTER_FD = -1


def main() -> int:
    global MASTER_FD
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="wit", choices=["wit", "csv"])
    ap.add_argument("--algo-deg", action="store_true", help="模拟以 deg/s 和 g 为单位输出的设备")
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--capture-seconds", type=float, default=12.0)
    args = ap.parse_args()

    master, slave = pty.openpty()
    MASTER_FD = master
    port = os.ttyname(slave)
    print(f"伪串口: {port}")
    print(f"模拟格式: {args.format}"
          + ("（原始单位 deg/s, g —— 用来验证单位换算）" if args.algo_deg else "（SI 单位）"))

    stop = threading.Event()
    feeder = threading.Thread(
        target=make_stream, args=(args.format, args.algo_deg, args.fs, args.duration, stop),
        daemon=True)
    feeder.start()

    scale = "deg/s,g" if args.algo_deg else "rad/s,m/s2"
    out = Path("/tmp/selftest_serial")
    cmd = [sys.executable, str(ROOT / "tools" / "capture_dataset.py"),
           "--out", str(out), "--no-gui",
           "--serial", port, "--baud", "115200",
           "--imu-format", args.format, "--imu-scale", scale]
    print("\n运行采集（无窗口模式，约 "
          f"{args.capture_seconds:.0f}s）:\n  " + " ".join(cmd) + "\n")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        time.sleep(args.capture_seconds)
    finally:
        # 用 SIGINT 而不是 SIGTERM：SIGINT 会变成 KeyboardInterrupt，
        # 采集脚本的 finally 才会执行并把缓冲写盘。
        # （采集脚本本身也装了 SIGTERM→KeyboardInterrupt 的转换，这里两种都能收尾。）
        proc.send_signal(signal.SIGINT)
        try:
            out_txt, _ = proc.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                out_txt, _ = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                out_txt, _ = proc.communicate()
        stop.set()
    print(out_txt)

    imu = out / "imu.txt"
    if not imu.exists() or imu.stat().st_size < 50:
        print("❌ 没写出 IMU 数据，解析链路有问题")
        return 1

    print("=" * 60)
    print("现在用体检工具检查这批合成数据（应当全部通过）：\n")
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "check_motion.py"),
                        "--imu", str(imu)], text=True)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
