#!/usr/bin/env python3
"""离线自测：**伪终端伪造 H7 IMU + 真实 UVC 相机**，把采集工具整条链路跑通。

为什么值得做
------------------------------------------------------------------
``tools/capture_h7.py`` 依赖两个硬件（工业相机 + H7 IMU），而这两个设备
不在手上时，**采集工具本身有没有 bug 完全测不出来** —— 直到真机接上那天，
你会在最不该出问题的地方浪费时间。

这个脚本用一个**伪终端（PTY）**冒充 H7 串口，按真实协议（82 字节帧 @1kHz，
``5A A5`` 帧头、含四元数/欧拉角/毫秒时间戳）持续吐数据；相机用本机 UVC
摄像头顶上。于是整条链路——串口打开、协议解析、后台线程、相机取流、
时间戳、图像落盘、深度多样性监测、meta 生成——全部走一遍。

真机一接上，剩下的唯一变量就只是"固件协议是否和解析一致"。

用法::

    .venv/bin/python tools/selftest_capture.py
    .venv/bin/python tools/selftest_capture.py --frames 5 --camera 0 --keep
"""
from __future__ import annotations

import argparse
import os
import pty
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calib"))

SYNC = b"\x5a\xa5"
PAYLOAD_LEN = 0x004C
FLOAT_NAMES = ("ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz",
               "roll", "pitch", "yaw", "qw", "qx", "qy", "qz")


def build_frame(t_ms: int, vals: dict) -> bytes:
    """按 imu_h7.py 记录的布局拼一帧（82 字节）。"""
    hdr = bytearray(14)
    hdr[0:2] = SYNC
    hdr[2:4] = PAYLOAD_LEN.to_bytes(2, "little")
    hdr[4:6] = (t_ms & 0xFFFF).to_bytes(2, "little")   # 该字段本机固件里是变化的
    hdr[6:8] = (0x0091).to_bytes(2, "little")
    hdr[8:10] = (0x002D).to_bytes(2, "little")
    body = struct.pack("<16f", *[vals[k] for k in FLOAT_NAMES])
    return bytes(hdr) + struct.pack("<I", t_ms & 0xFFFFFFFF) + body


class FakeH7:
    """在 PTY 主端按 1 kHz 写 H7 帧；从端就是 capture_h7 会打开的"串口"。"""

    def __init__(self, rate: float = 1000.0, motion: bool = True):
        self.master, self.slave = pty.openpty()
        self.slave_name = os.ttyname(self.slave)
        self.rate = rate
        self.motion = motion
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self.n_written = 0

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=1.0)
        for fd in (self.master, self.slave):
            try:
                os.close(fd)
            except OSError:
                pass

    def _run(self):
        t0 = time.monotonic()
        k = 0
        while not self._stop.is_set():
            target = t0 + k / self.rate
            dt = target - time.monotonic()
            if dt > 0:
                time.sleep(min(dt, 0.002))
            t = k / self.rate
            w = 0.8 if self.motion else 0.0
            vals = {
                "ax": 0.05 * w * np.sin(3 * t), "ay": 0.04 * w * np.sin(2 * t),
                "az": 9.80665, "gx": 0.7 * w * np.sin(2.6 * t),
                "gy": 0.5 * w * np.sin(3.6 * t), "gz": 0.3 * w * np.sin(1.5 * t),
                "mx": 0.0, "my": 0.0, "mz": 0.0,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
            }
            try:
                os.write(self.master, build_frame(1000 + k, vals))
                self.n_written += 1
            except OSError:
                break
            k += 1


def main() -> int:
    ap = argparse.ArgumentParser(description="采集工具离线整链路自测")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--camera", default="0", help="UVC 索引（本机 /dev/video0）")
    ap.add_argument("--out", default="/tmp/selftest_capture")
    ap.add_argument("--keep", action="store_true", help="保留输出目录")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    print("=" * 66)
    print("采集工具离线整链路自测（伪 IMU + 真实 UVC 相机）")
    print("=" * 66)

    imu = FakeH7()
    print(f"① 伪 H7 串口已建立: {imu.slave_name}")
    imu.start()
    time.sleep(0.3)
    print(f"   已写入 {imu.n_written} 帧（协议：82 字节 @1kHz）")

    cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "tools/capture_h7.py"),
           "--out", str(out), "--serial", imu.slave_name,
           "--camera-source", args.camera, "--width", "640", "--height", "480",
           "--frames", str(args.frames), "--frame-interval", "0.1",
           "--pattern", "9x6", "--square-mm", "25"]
    print(f"\n② 运行采集工具（相机 index={args.camera}，抓 {args.frames} 帧）...")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    imu.stop()

    for ln in r.stdout.strip().splitlines():
        print("   | " + ln)
    if r.returncode != 0:
        print(f"\n❌ 采集工具退出码 {r.returncode}")
        print(r.stderr[-1500:])
        return 1

    print("\n③ 校验产出")
    checks = []
    imgs = sorted((out / "images").glob("*.png"))
    checks.append(("图像数量", len(imgs) == args.frames, f"{len(imgs)}/{args.frames}"))

    ts = out / "image_timestamps.txt"
    ok = ts.exists()
    n_ts = 0
    if ok:
        lines = [l for l in ts.read_text().splitlines() if l and not l.startswith("#")]
        n_ts = len(lines)
        ok = n_ts == args.frames
    checks.append(("image_timestamps.txt", ok, f"{n_ts} 行"))

    imu_f = out / "imu.txt"
    n_imu = 0
    ok = imu_f.exists()
    if ok:
        lines = [l for l in imu_f.read_text().splitlines() if l and not l.startswith("#")]
        n_imu = len(lines)
        ok = n_imu > 50                      # 采集几秒，至少要上百帧
    checks.append(("imu.txt", ok, f"{n_imu} 帧"))

    checks.append(("imu_h7_full.csv", (out / "imu_h7_full.csv").exists(), ""))
    checks.append(("imu_raw.bin", (out / "imu_raw.bin").exists(), ""))
    checks.append(("camera_timestamps.csv", (out / "camera_timestamps.csv").exists(), ""))
    meta = out / "meta.json"
    checks.append(("meta.json", meta.exists(), ""))

    # imu.txt 第一列必须是单调递增的
    mono = False
    if n_imu > 10:
        t = np.array([float(l.split()[0]) for l in
                      imu_f.read_text().splitlines() if l and not l.startswith("#")])
        mono = bool(np.all(np.diff(t) > 0))
    checks.append(("imu 时间戳单调递增", mono, ""))

    # 用我们自己的一键工具读一遍这份数据的前半段（证明格式自洽）
    # 这里只验读取，不做完整标定（相机没对着板子）
    readable = False
    try:
        from io_data import load_imu_txt, load_image_timestamps
        d = load_imu_txt(imu_f)
        m = load_image_timestamps(ts)
        readable = len(d) == n_imu and len(m) == args.frames
    except Exception as e:  # noqa: BLE001
        print(f"   ⚠️ 读取失败: {e}")
    checks.append(("能被 io_data 重新读回", readable, ""))

    ok_all = True
    for name, good, note in checks:
        print(f"   {'✅' if good else '❌'} {name:26s} {note}")
        ok_all &= good

    if not args.keep:
        shutil.rmtree(out, ignore_errors=True)

    print("\n" + "=" * 66)
    print("✅ 采集链路完整可用（真机接上后只剩'协议是否匹配'一个变量）"
          if ok_all else "❌ 有未通过项，见上面标 ❌ 的行")
    print("=" * 66)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
