#!/usr/bin/env python3
"""统一相机抽象层：海康 U3V 工业相机（aravis）与普通 UVC 摄像头（OpenCV）。

为什么要有这一层
------------------------------------------------------------------
标定算法不应该关心图像是从哪来的。这个模块把两种完全不同的取流方式
（USB3 Vision 走 GenICam，UVC 走 V4L2）包装成同一个接口：

    with open_camera("hik") as cam:
        cam.configure(width=1624, height=1240, pixel_format="Mono8",
                      exposure_us=8000, gain_db=10, frame_rate=10)
        for frame in cam.frames(5):
            frame.image   # np.ndarray
            frame.device_ts / frame.host_ts

关于时间戳（C2 阶段的关键）
------------------------------------------------------------------
每个 :class:`Frame` 带两个时间戳：

* ``host_ts``  —— 主机**单调时钟**（秒，``time.monotonic()`` 基准）。
  相机时间戳与 IMU 时间戳必须换算到同一个时钟才能对齐，单调钟不会因 NTP
  校时跳变，是唯一安全的选择。
* ``device_ts`` —— 相机**自己的硬件时间戳**（MV-CS020-10UC 支持
  ``DeviceTimestamp`` 特性）。它不随主机负载抖动，比主机接收时刻干净得多。

U3V 分支的做法：aravis 在收帧那一刻会把**墙钟**（ns）写进 buffer，这比我们
从 buffer 队列里 pop 出来的时刻更早也更准。所以本模块在打开相机时记录一组
（墙钟, 单调钟）基准，把 aravis 的墙钟时间戳换算成单调钟，与 ``host_ts``
互相校验。

关于像素格式：**标定请用 Mono8**
------------------------------------------------------------------
这块相机是彩色传感器，但实测 ``Mono8`` 在**相机内部**完成灰度转换，
不是把原始 Bayer 拼图直接丢出来（实测对角差 1.26 vs 轴向差 1.07，
若是 Bayer 拼图会出现 2~3 倍的高频棋盘格伪影，会严重伤害亚像素角点检测）。

依赖
------------------------------------------------------------------
U3V 分支需要 aravis 的 Python 绑定（GObject Introspection）：:

    sudo apt install -y gir1.2-aravis-0.8      # 只要 29.7 kB

没有 sudo 时的临时办法（仅本次会话有效，也是本仓开发时用的方式）：:

    cd /tmp && apt-get download gir1.2-aravis-0.8 && dpkg -x gir1.2-aravis-0.8_*.deb ex
    export GI_TYPELIB_PATH=/tmp/ex/usr/lib/x86_64-linux-gnu/girepository-1.0

UVC 分支只需要 OpenCV。
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np

# ── 常用像素格式 -> numpy 形状 ────────────────────────────────
_MONO_FORMATS = {"Mono8"}
_BAYER_FORMATS = {"BayerRG8", "BayerGR8", "BayerGB8", "BayerBG8"}
_BGR_FORMATS = {"BGR8Packed"}
_RGB_FORMATS = {"RGB8Packed"}


@dataclass
class Frame:
    """一帧图像 + 它的两个时间戳。"""

    image: np.ndarray
    host_ts: float           # 主机单调时钟，秒
    device_ts: int | None    # 相机硬件时间戳（tick），None = 不支持
    frame_id: int = 0

    @property
    def shape(self):
        return self.image.shape


# ══════════════════════════════════════════════════════════════
#  海康 U3V 工业相机（aravis）
# ══════════════════════════════════════════════════════════════
def _import_aravis():
    """导入 aravis 的 Python 绑定，失败时给出可执行的修复命令。"""
    try:
        import gi

        gi.require_version("Aravis", "0.8")
        from gi.repository import Aravis

        return Aravis
    except (ImportError, ValueError) as e:  # noqa: BLE001
        raise RuntimeError(
            "用不了 aravis 的 Python 绑定。装一下（29.7 kB）：\n"
            "    sudo apt install -y gir1.2-aravis-0.8\n"
            "没有 sudo 时可以先临时抽出 typelib：\n"
            "    cd /tmp && apt-get download gir1.2-aravis-0.8\n"
            "    dpkg -x gir1.2-aravis-0.8_*.deb ex\n"
            "    export GI_TYPELIB_PATH=/tmp/ex/usr/lib/x86_64-linux-gnu/girepository-1.0"
        ) from e


def list_devices() -> list[dict]:
    """列出 aravis 能看到的所有 GenICam 设备（U3V + GigE）。"""
    Aravis = _import_aravis()
    Aravis.update_device_list()
    out = []
    for i in range(Aravis.get_n_devices()):
        out.append({
            "index": i,
            "id": Aravis.get_device_id(i),
            "vendor": Aravis.get_device_vendor(i),
            "model": Aravis.get_device_model(i),
            "serial": Aravis.get_device_serial_nbr(i),
            "protocol": Aravis.get_device_protocol(i),
        })
    return out


class HikCamera:
    """USB3 Vision / GigE 工业相机。接口与 :class:`UvcCamera` 一致。"""

    kind = "hik"

    def __init__(self, device_index: int = 0, device_id: str | None = None):
        self.Aravis = _import_aravis()
        A = self.Aravis
        A.update_device_list()
        if A.get_n_devices() == 0:
            raise RuntimeError("aravis 没发现任何相机。检查：lsusb | grep 2bdf；"
                               "确认相机插在 USB3 口上（lsusb -t 应显示 5000M）")
        self.device_id = device_id or A.get_device_id(device_index)
        self.cam = A.Camera.new(self.device_id)
        self.stream = None
        self._n_buffers = 0
        self._payload = 0
        self._ring: list = []          # 待回收的 buffer，避免频繁分配
        # 时间戳换算基准：aravis 的 system timestamp 是墙钟(ns)，换成单调钟
        self._mono0 = time.monotonic()
        self._wall0 = time.time()
        self.n_missing = 0
        self.n_ok = 0
        self.timestamp_tick_hz = self._read_tick_hz()

    # -- 内部 --------------------------------------------------
    def _read_tick_hz(self):
        """设备时间戳的 tick 频率（把 device_ts 换算成秒要用）。"""
        c = self.cam
        for feat in ("TimestampTickFrequency", "GevTimestampTickFrequency"):
            try:
                if c.is_feature_available(feat):
                    v = c.get_integer(feat)
                    if v > 0:
                        return v
            except Exception:  # noqa: BLE001
                pass
        # U3V 规范要求 1 GHz；实测这块相机 device_ts 增长也符合 ns 量级
        return 1_000_000_000

    def _wall_ns_to_mono(self, ns: int) -> float:
        if ns <= 0:
            return time.monotonic()
        mono = self._mono0 + (ns / 1e9 - self._wall0)
        # 墙钟若被 NTP 校时，换算值可能离谱；越界就退回当前单调钟
        now = time.monotonic()
        if abs(mono - now) > 1.0:
            return now
        return mono

    # -- 生命周期 ----------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            if self.stream is not None:
                self.cam.stop_acquisition()
        except Exception:  # noqa: BLE001
            pass
        self.stream = None
        self._ring.clear()
        # 必须显式释放：Camera 持有 USB 控制接口，不回收的话下次打开会
        # 报 LIBUSB_ERROR_BUSY。实测 del + gc.collect() 就够了。
        self.cam = None
        gc.collect()
        time.sleep(0.15)

    # -- 信息 --------------------------------------------------
    def info(self) -> dict:
        c = self.cam
        d = {
            "kind": self.kind,
            "device_id": self.device_id,
            "vendor": c.get_vendor_name(),
            "model": c.get_model_name(),
            "serial": c.get_device_serial_number(),
        }
        for feat, key in (("Width", "width"), ("Height", "height")):
            try:
                d[key] = c.get_integer(feat)
            except Exception:  # noqa: BLE001
                pass
        try:
            d["pixel_format"] = c.get_pixel_format_as_string()
        except Exception:  # noqa: BLE001
            pass
        try:
            d["width_bounds"] = tuple(c.get_width_bounds())
            d["height_bounds"] = tuple(c.get_height_bounds())
        except Exception:  # noqa: BLE001
            pass
        try:
            d["pixel_formats"] = list(c.dup_available_pixel_formats_as_strings())
        except Exception:  # noqa: BLE001
            pass
        try:
            d["exposure_us_bounds"] = tuple(c.get_exposure_time_bounds())
            d["gain_db_bounds"] = tuple(c.get_gain_bounds())
            d["frame_rate_bounds"] = tuple(c.get_frame_rate_bounds())
        except Exception:  # noqa: BLE001
            pass
        d["device_timestamp_supported"] = bool(
            c.is_feature_available("DeviceTimestamp")
        )
        d["timestamp_tick_hz"] = self.timestamp_tick_hz
        return d

    # -- 配置 --------------------------------------------------
    def configure(self, width=None, height=None, pixel_format="Mono8",
                  exposure_us=None, exposure_auto=False, gain_db=None,
                  frame_rate=None, n_buffers=8):
        """设置采集参数。所有项都是可选的，只改传进来的。"""
        A, c = self.Aravis, self.cam

        # 改分辨率/像素格式前必须先停流，否则会报错
        if self.stream is not None:
            self.close()
            self.cam = A.Camera.new(self.device_id)

        if pixel_format:
            avail = list(c.dup_available_pixel_formats_as_strings())
            if pixel_format not in avail:
                raise ValueError(f"相机不支持 {pixel_format}，可选：{avail}")
            c.set_pixel_format_from_string(pixel_format)
        if width:
            c.set_integer("Width", int(width))
        if height:
            c.set_integer("Height", int(height))

        if exposure_auto:
            c.set_exposure_time_auto(A.Auto.CONTINUOUS)
        else:
            c.set_exposure_time_auto(A.Auto.OFF)
            if exposure_us is not None:
                c.set_exposure_time(float(exposure_us))
        if gain_db is not None:
            c.set_gain_auto(A.Auto.OFF)
            c.set_gain(float(gain_db))
        if frame_rate is not None:
            # 必须先打开帧率控制开关，否则 set_frame_rate 会被相机忽略
            # （arv-test-0.8 报 "FrameRate FAILURE 179.99Hz (expected 10Hz)" 就是这个原因）
            c.set_frame_rate_enable(True)
            c.set_frame_rate(float(frame_rate))

        self._payload = c.get_payload()
        self._n_buffers = n_buffers
        self.stream = c.create_stream(None, None, n_buffers)
        for _ in range(n_buffers):
            self.stream.push_buffer(A.Buffer.new_allocate(self._payload))
        return self

    # -- 取流 --------------------------------------------------
    def start(self):
        if self.stream is None:
            self.configure()
        self.cam.start_acquisition()

    def stop(self):
        if self.stream is not None:
            self.cam.stop_acquisition()

    def _decode(self, buf) -> np.ndarray:
        w, h = buf.get_image_width(), buf.get_image_height()
        raw = np.frombuffer(buf.get_data(), dtype=np.uint8)
        pf = self.cam.get_pixel_format_as_string()
        if pf in _MONO_FORMATS:
            return raw[: w * h].reshape(h, w)
        if pf in _BAYER_FORMATS:
            return raw[: w * h].reshape(h, w)
        if pf in _RGB_FORMATS or pf in _BGR_FORMATS:
            return raw[: w * h * 3].reshape(h, w, 3)
        return raw[: w * h].reshape(h, w)

    def frames(self, n: int | None = None, timeout_us: int = 2_000_000
               ) -> Iterator[Frame]:
        """产出帧。``n=None`` 表示一直取（调用方负责 break）。"""
        A = self.Aravis
        if self.stream is None:
            self.configure()
        started_here = False
        if self.cam.get_acquisition_mode() != A.AcquisitionMode.CONTINUOUS:
            self.cam.set_acquisition_mode(A.AcquisitionMode.CONTINUOUS)
        try:
            self.cam.start_acquisition()
            started_here = True
        except Exception:  # noqa: BLE001
            pass  # 已经在采了

        k = 0
        try:
            while n is None or k < n:
                buf = self.stream.timeout_pop_buffer(timeout_us)
                if buf is None:
                    raise TimeoutError(
                        f"取帧超时（{timeout_us/1e6:.1f}s）。可能原因：相机被别的"
                        f"进程占用、USB 链路不稳、或曝光+帧率组合超过带宽。"
                    )
                status = buf.get_status()
                if status == A.BufferStatus.SUCCESS:
                    self.n_ok += 1
                    yield Frame(
                        image=self._decode(buf),
                        host_ts=self._wall_ns_to_mono(buf.get_system_timestamp()),
                        device_ts=buf.get_timestamp() or None,
                        frame_id=k,
                    )
                    k += 1
                else:
                    self.n_missing += 1
                # 归还缓冲，保持流水线不断
                self.stream.push_buffer(A.Buffer.new_allocate(self._payload))
        finally:
            if started_here:
                try:
                    self.cam.stop_acquisition()
                except Exception:  # noqa: BLE001
                    pass


# ══════════════════════════════════════════════════════════════
#  普通 UVC 摄像头（OpenCV）
# ══════════════════════════════════════════════════════════════
class UvcCamera:
    """UVC 摄像头（笔记本自带那种）。算法开发阶段的默认数据源。"""

    kind = "uvc"

    def __init__(self, index: int = 0, **_):
        import cv2

        self.cv2 = cv2
        self.index = index
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(
                f"打不开 UVC 相机 index={index}。若这是海康工业相机，它**不是 UVC**，"
                f"请用 --camera-source hik。"
            )
        self.n_ok = 0
        self.n_missing = 0
        self.timestamp_tick_hz = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            self.cap.release()
        except Exception:  # noqa: BLE001
            pass

    def info(self) -> dict:
        return {
            "kind": self.kind,
            "index": self.index,
            "width": int(self.cap.get(3)),
            "height": int(self.cap.get(4)),
            "fps": float(self.cap.get(5)),
            "device_timestamp_supported": False,
        }

    def configure(self, width=None, height=None, pixel_format=None,
                  exposure_us=None, exposure_auto=False, gain_db=None,
                  frame_rate=None, n_buffers=1):
        # 把驱动内部缓冲压到 1 帧：默认缓冲会让"读到帧的时刻"比曝光时刻晚
        # 几十毫秒且不稳定，这是时间对齐最常见的精度杀手。
        self.cap.set(self.cv2.CAP_PROP_BUFFERSIZE, 1)
        if width:
            self.cap.set(self.cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height:
            self.cap.set(self.cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if frame_rate:
            self.cap.set(self.cv2.CAP_PROP_FPS, float(frame_rate))
        if exposure_us is not None and not exposure_auto:
            # V4L2 语义：AUTO_EXPOSURE=1 手动、3 自动。实现随驱动而异，失败不致命
            self.cap.set(self.cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.cap.set(self.cv2.CAP_PROP_EXPOSURE, float(exposure_us))
        return self

    def start(self):
        pass

    def stop(self):
        pass

    def frames(self, n=None, timeout_us=2_000_000) -> Iterator[Frame]:
        k = 0
        while n is None or k < n:
            ok, img = self.cap.read()
            t = time.monotonic()      # 紧跟 read() 之后取时刻，越早越准
            if not ok:
                self.n_missing += 1
                raise TimeoutError("UVC 读帧失败")
            self.n_ok += 1
            gray = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY)
            yield Frame(image=gray, host_ts=t, device_ts=None, frame_id=k)
            k += 1


# ══════════════════════════════════════════════════════════════
#  工厂
# ══════════════════════════════════════════════════════════════
def open_camera(source="uvc", index=0, **kw):
    """``source``: ``"hik"`` → 工业相机；``"uvc"`` 或数字字符串 → 普通摄像头。"""
    s = str(source).lower()
    if s in ("hik", "u3v", "genicam", "aravis"):
        return HikCamera(device_index=index)
    return UvcCamera(index=int(source) if s.isdigit() else index)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="相机自检 / 取图（U3V 工业相机 + UVC 摄像头）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--list", action="store_true", help="列出 aravis 能看到的设备")
    ap.add_argument("--source", default="hik", help="hik 或 0/1/... (UVC 索引)")
    ap.add_argument("--info", action="store_true", help="打印相机能力与参数范围")
    ap.add_argument("--grab", type=int, default=0, help="抓 N 帧并保存")
    ap.add_argument("--out", default="/tmp/cam", help="保存目录")
    ap.add_argument("--width", type=int, default=1624)
    ap.add_argument("--height", type=int, default=1240)
    ap.add_argument("--pixel-format", default="Mono8")
    ap.add_argument("--exposure", type=float, default=None, help="曝光, 微秒")
    ap.add_argument("--gain", type=float, default=None, help="增益, dB")
    ap.add_argument("--frame-rate", type=float, default=10.0)
    args = ap.parse_args()

    if args.list:
        devs = list_devices()
        print(f"aravis 发现 {len(devs)} 个设备")
        for d in devs:
            print(f"  [{d['index']}] {d['protocol']}:{d['model']} "
                  f"SN={d['serial']}  id={d['id']}")
        return 0

    with open_camera(args.source) as cam:
        info = cam.info()
        print("=== 相机信息 ===")
        for k, v in info.items():
            print(f"  {k:28s} {v}")

        if args.info and args.grab == 0:
            return 0

        cam.configure(width=args.width, height=args.height,
                      pixel_format=args.pixel_format,
                      exposure_us=args.exposure, gain_db=args.gain,
                      frame_rate=args.frame_rate)
        n = args.grab or 3
        from pathlib import Path

        import cv2

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== 抓 {n} 帧 -> {out} ===")
        t0 = time.monotonic()
        prev_dev = None
        for f in cam.frames(n):
            name = out / f"{f.frame_id:06d}.png"
            cv2.imwrite(str(name), f.image)
            d = ""
            if f.device_ts and prev_dev:
                dt = (f.device_ts - prev_dev) / cam.timestamp_tick_hz
                d = f"  设备间隔={dt*1e3:7.2f} ms"
            prev_dev = f.device_ts
            print(f"  [{f.frame_id}] {f.image.shape} dtype={f.image.dtype} "
                  f"mean={f.image.mean():6.1f} host_ts={f.host_ts:.3f} "
                  f"device_ts={f.device_ts}{d}")
        dt = time.monotonic() - t0
        print(f"\n实际帧率 ≈ {n/dt:.2f} Hz；丢包帧 {cam.n_missing}，成功 {cam.n_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
