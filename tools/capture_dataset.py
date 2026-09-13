#!/usr/bin/env python3
"""数据采集：相机图像 + IMU 记录，带统一时钟时间戳。

用法:
    # 只用相机（现在就能跑）
    .venv/bin/python tools/capture_dataset.py --out data/session_01 --camera 0

    # 相机 + 串口 IMU
    .venv/bin/python tools/capture_dataset.py --out data/session_01 --camera 0 \
        --serial /dev/ttyUSB0 --baud 921600 --imu-format wit

    # 先看看 IMU 吐什么，不落盘
    .venv/bin/python tools/capture_dataset.py --probe-serial /dev/ttyUSB0 --baud 921600

按键（窗口获得焦点时）:
    s / 空格   保存当前帧
    a          自动连拍（每 0.5s 一帧），再按一次停止
    q / ESC    结束采集并写盘

为什么不用 imencode 直接写视频：标定要的是**每帧独立的时间戳**，
所以这里写成逐帧 PNG + 一个 timestamps 文本，任务文档 2.1 节要求如此。
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MONO = time.monotonic  # 跨设备对齐用单调时钟；系统时钟会被 NTP 跳变污染


def _install_signal_handlers() -> None:
    """把 SIGTERM/SIGINT 转成 KeyboardInterrupt，保证收尾逻辑一定跑到。

    默认情况下 SIGTERM 直接终止进程，finally 不会执行 —— 那样采到的数据会全丢。
    无人值守或脚本里 kill 掉采集时，这个转换是关键。
    """
    def handler(signum, _frame):
        raise KeyboardInterrupt(f"收到信号 {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass  # 非主线程或平台不支持时忽略


# ─────────────────────────── IMU 串口解析 ───────────────────────────
@dataclass
class ImuSample:
    t: float
    gx: float
    gy: float
    gz: float
    ax: float
    ay: float
    az: float


class SerialImu:
    """按"正则 + 字段顺序"解析任意文本协议的 IMU。

    常见协议已经内置（--imu-format），未知协议用 --imu-regex 传自定义正则，
    --imu-scale 传单位换算（例如把 g 换成 m/s^2、把 deg/s 换成 rad/s）。
    """

    PRESETS = {
        # 维特/WIT 标准模式: 0x55 0x51 ... 二进制。这里给的是常见的 ASCII 输出
        "wit": r"t=(?P<t>[-\d.]+).*?gx=(?P<gx>[-\d.]+).*?gy=(?P<gy>[-\d.]+).*?gz=(?P<gz>[-\d.]+).*?ax=(?P<ax>[-\d.]+).*?ay=(?P<ay>[-\d.]+).*?az=(?P<az>[-\d.]+)",
        # 纯数字列，默认认为已经是 SI 单位
        "csv": r"^[^-\d]*(?P<t>[-\d.]+)[,\s]+(?P<gx>[-\d.]+)[,\s]+(?P<gy>[-\d.]+)[,\s]+(?P<gz>[-\d.]+)[,\s]+(?P<ax>[-\d.]+)[,\s]+(?P<ay>[-\d.]+)[,\s]+(?P<az>[-\d.]+)",
    }

    def __init__(self, port: str, baud: int, fmt: str, regex: str | None, scale: str,
                 use_local_time: bool = True):
        import serial  # 延迟导入，未装 pyserial 时相机模式仍可用

        self.ser = serial.Serial(port, baud, timeout=0.05)
        self.rx = getattr(self.ser, "in_waiting", 0)
        self.pattern = re.compile(regex or self.PRESETS.get(fmt, self.PRESETS["csv"]))
        self.use_local_time = use_local_time
        self.scale = self._parse_scale(scale)
        self.samples: list[ImuSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.bad_lines = 0
        self.good_lines = 0
        self.first_raw: list[str] = []

    @staticmethod
    def _parse_scale(scale: str) -> dict[str, float]:
        """--imu-scale deg/s,g  →  {gyro: pi/180, accel: 9.81}"""
        out = {"gyro": 1.0, "accel": 1.0}
        for part in scale.split(","):
            part = part.strip().lower()
            if part in ("deg/s", "dps", "deg"):
                out["gyro"] = np.pi / 180.0
            elif part in ("rad/s", "rad"):
                out["gyro"] = 1.0
            elif part in ("g",):
                out["accel"] = 9.81
            elif part in ("m/s2", "m/s^2"):
                out["accel"] = 1.0
        return out

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # 设备时间戳经常与本地单调时钟差一个巨大常数，先攒样本再判定
        while not self._stop.is_set():
            try:
                raw = self.ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception as exc:  # noqa: BLE001
                print(f"[IMU] 读取异常: {exc}")
                break
            if not raw:
                continue
            if len(self.first_raw) < 5:
                self.first_raw.append(raw)
            m = self.pattern.search(raw)
            if not m:
                self.bad_lines += 1
                continue
            d = m.groupdict()
            try:
                t = float(d.get("t") or 0.0)
                if self.use_local_time or t == 0.0:
                    t = MONO()
                s = ImuSample(
                    t=t,
                    gx=float(d["gx"]) * self.scale["gyro"],
                    gy=float(d["gy"]) * self.scale["gyro"],
                    gz=float(d["gz"]) * self.scale["gyro"],
                    ax=float(d["ax"]) * self.scale["accel"],
                    ay=float(d["ay"]) * self.scale["accel"],
                    az=float(d["az"]) * self.scale["accel"],
                )
            except (KeyError, ValueError, TypeError):
                self.bad_lines += 1
                continue
            self.good_lines += 1
            self.samples.append(s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self.ser.close()

    def drain(self) -> list[ImuSample]:
        """取出并清空当前缓冲（采集循环里定期调用，避免内存无限增长）。"""
        out, self.samples = self.samples, []
        return out


def probe_serial(port: str, baud: int, seconds: float) -> int:
    """先看设备到底吐什么格式，不落盘。"""
    import serial

    print(f"打开 {port} @ {baud}，监听 {seconds:.0f} 秒原始输出…\n")
    try:
        ser = serial.Serial(port, baud, timeout=0.2)
    except Exception as exc:  # noqa: BLE001
        print(f"打不开串口: {exc}")
        print("检查: 设备是否插好 / 是否有权限 (ls -l " + port + ") / 波特率是否正确")
        return 1
    t0 = MONO()
    n = 0
    while MONO() - t0 < seconds:
        raw = ser.readline()
        if not raw:
            continue
        n += 1
        if n <= 15:
            try:
                txt = raw.decode("utf-8", errors="replace").rstrip()
            except Exception:  # noqa: BLE001
                txt = repr(raw)
            print(f"  [{n:3d}] {txt}")
        elif raw[:1] == b"\x55":
            pass  # 二进制协议，跳过打印
    print(f"\n{n} 行 / {seconds:.0f}s  →  平均 {n / seconds:.1f} Hz")
    if n == 0:
        print("一行都没读到：波特率大概率不对，或设备需要先发启动指令。")
    ser.close()
    return 0


# ─────────────────────────── 采集主循环 ───────────────────────────
@dataclass
class Session:
    out: Path
    frames: list[tuple[str, float]] = field(default_factory=list)
    imu: list[ImuSample] = field(default_factory=list)

    def write(self) -> None:
        (self.out / "images").mkdir(parents=True, exist_ok=True)
        with (self.out / "image_timestamps.txt").open("w") as f:
            f.write("# filename timestamp_seconds (monotonic clock, unit: s)\n")
            for name, t in self.frames:
                f.write(f"{name} {t:.9f}\n")
        with (self.out / "imu.txt").open("w") as f:
            f.write("# timestamp gx gy gz ax ay az  (s, rad/s, rad/s, rad/s, m/s^2 ...)\n")
            for s in self.imu:
                f.write(f"{s.t:.9f} {s.gx:.9f} {s.gy:.9f} {s.gz:.9f} "
                        f"{s.ax:.9f} {s.ay:.9f} {s.az:.9f}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="采集 Camera-IMU 标定数据")
    ap.add_argument("--out", default="data/session_01", help="输出目录")
    ap.add_argument("--camera", type=int, default=0, help="相机索引或 /dev/videoN 的 N")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--serial", default=None, help="IMU 串口，如 /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--imu-format", default="wit", choices=sorted(SerialImu.PRESETS))
    ap.add_argument("--imu-regex", default=None, help="自定义正则，需含命名组 t,gx,gy,gz,ax,ay,az")
    ap.add_argument("--imu-scale", default="rad/s,m/s2",
                    help="原始单位，逗号分隔：deg/s 或 rad/s；g 或 m/s2")
    ap.add_argument("--imu-local-time", action="store_true", default=True,
                    help="忽略设备自带时间戳，统一用本机单调时钟（默认开）")
    ap.add_argument("--probe-serial", default=None, metavar="PORT",
                    help="只探测串口输出格式，不采集")
    ap.add_argument("--probe-seconds", type=float, default=5.0)
    ap.add_argument("--no-gui", action="store_true", help="无窗口模式：只采集 IMU，不抓图")
    ap.add_argument("--frames", type=int, default=0,
                    help="自测模式：不开窗口，自动抓 N 帧后退出（用于验证整条链路）")
    args = ap.parse_args()

    _install_signal_handlers()

    if args.probe_serial:
        return probe_serial(args.probe_serial, args.baud, args.probe_seconds)

    out = Path(os.path.expanduser(args.out))
    sess = Session(out=out)

    imu: SerialImu | None = None
    if args.serial:
        try:
            imu = SerialImu(args.serial, args.baud, args.imu_format,
                            args.imu_regex, args.imu_scale, args.imu_local_time)
            imu.start()
            print(f"[IMU] {args.serial} @ {args.baud} 已开始接收")
        except Exception as exc:  # noqa: BLE001
            print(f"[IMU] 打开失败: {exc}")
            print("     改用纯相机模式继续。先跑 --probe-serial 排查设备与权限。")
            imu = None

    cap = None
    if not args.no_gui:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        if not cap.isOpened():
            print(f"相机 {args.camera} 打不开；用 --no-gui 可以只采 IMU")
            if imu:
                imu.stop()
            return 1
        print(f"[相机] video{args.camera} {args.width}x{args.height}@{args.fps}")

    if args.frames > 0:
        # 自测模式：不开窗口，抓 N 帧后自动退出，其余走同一套收尾逻辑
        if cap is None:
            print("--frames 需要相机；若只想收 IMU 请用 --no-gui")
            if imu:
                imu.stop()
            return 1
        print(f"[自测] 抓 {args.frames} 帧，每帧间隔 0.2s …")
        for i in range(args.frames):
            ok, frame = cap.read()
            if not ok:
                print(f"  第 {i + 1} 帧读取失败")
                break
            t = MONO()
            name = f"{i:06d}.png"
            (out / "images").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "images" / name), frame)
            sess.frames.append((name, t))
            if imu is not None:
                sess.imu.extend(imu.drain())
            time.sleep(0.2)
        print(f"[自测] 完成，抓到 {len(sess.frames)} 帧")
        if imu is not None:
            sess.imu.extend(imu.drain())
            imu.stop()
            print(f"[IMU] 有效 {imu.good_lines} 行 / 丢弃 {imu.bad_lines} 行")
            imu = None
        cap.release()
        sess.write()
        print(f"\n写入 {out}")
        print(f"  图像     {len(sess.frames)} 帧 → {out / 'images'}")
        print(f"  图像时间戳 → {out / 'image_timestamps.txt'}")
        print(f"  IMU      {len(sess.imu)} 条 → {out / 'imu.txt'}")
        return 0

    print("\n按键: s/空格=存帧  a=自动连拍(0.5s)  q/ESC=结束\n")
    auto = False
    last_auto = 0.0
    idx = 0
    t_start = MONO()

    try:
        while True:
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    print("读帧失败，退出"); break
                if imu is not None:
                    sess.imu.extend(imu.drain())
                    # 把最近一条 IMU 的时间对齐到这一帧，便于事后检查同步
                    sync = sess.imu[-1].t - MONO() if sess.imu else 0.0
                else:
                    sync = 0.0
                view = frame.copy()
                txt = (f"frames={len(sess.frames)} imu={len(sess.imu)} "
                       f"auto={'ON' if auto else 'off'} sync_off={sync * 1e3:+.0f}ms")
                cv2.putText(view, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow("capture  s=save a=auto q=quit", view)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255
                time.sleep(0.01)
                if imu is not None:
                    sess.imu.extend(imu.drain())

            now = MONO()
            if auto and now - last_auto >= 0.5:
                key = ord("s")
                last_auto = now

            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                auto = not auto
                print(f"自动连拍: {'开' if auto else '关'}")
            if key in (ord("s"), ord(" ")):
                t = MONO()
                name = f"{idx:06d}.png"
                (out / "images").mkdir(parents=True, exist_ok=True)
                if cap is not None:
                    cv2.imwrite(str(out / "images" / name), frame)
                sess.frames.append((name, t))
                idx += 1
                if idx % 10 == 0 or idx <= 3:
                    print(f"  已存 {idx} 帧  (t={t - t_start:.2f}s)")
    except KeyboardInterrupt:
        print("\n收到中断，收尾…")
    finally:
        if imu is not None:
            time.sleep(0.1)
            sess.imu.extend(imu.drain())
            imu.stop()
            print(f"[IMU] 有效 {imu.good_lines} 行 / 丢弃 {imu.bad_lines} 行")
            if imu.good_lines == 0 and imu.first_raw:
                print("[IMU] 没解析出任何样本，设备原始输出前几行是：")
                for r in imu.first_raw:
                    print("      ", r[:120])
                print("      用 --imu-regex 自定义正则，或 --imu-format csv，或确认波特率")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        sess.write()
        print(f"\n写入 {out}")
        print(f"  图像     {len(sess.frames)} 帧 → {out / 'images'}")
        print(f"  图像时间戳 → {out / 'image_timestamps.txt'}")
        print(f"  IMU      {len(sess.imu)} 条 → {out / 'imu.txt'}")
        if sess.imu:
            dur = sess.imu[-1].t - sess.imu[0].t
            print(f"  IMU 频率 ≈ {len(sess.imu) / max(dur, 1e-6):.0f} Hz，时长 {dur:.1f}s")
            print("\n下一步: .venv/bin/python tools/check_motion.py --imu "
                  + str(out / "imu.txt") + "   ← 检查运动激励够不够做标定")
    return 0


if __name__ == "__main__":
    sys.exit(main())
