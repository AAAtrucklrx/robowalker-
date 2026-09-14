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
import os
import time
from dataclasses import dataclass
from pathlib import Path
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


# ══════════════════════════════════════════════════════════════
#  USB 假死自愈
# ══════════════════════════════════════════════════════════════
# 2026-09-14 实测确认的规律（这是本模块最值得记住的一条）：
#
#   连续 5 次「打开→配置→取帧→**正常 close()**」循环 → 全部成功；
#   只要有一次**不干净释放**（进程被强杀 / SIGABRT / 忘了 close）
#   → 下一次就在 ``set_pixel_format_from_string`` 报
#     ``[PixelFormat_Reg] USB3Vision write_memory error``，
#     之后永远 ``LIBUSB_ERROR_BUSY``，所有取景工具都是**黑屏**。
#
# 也就是说"相机黑屏"九成不是相机坏了，而是上一次没干净退出。
# 好消息是：``/dev/bus/usb/...`` 在本机是 777，一次 ``USBDEVFS_RESET``
# ioctl 就能救回来，**不需要 root、不需要拔插**。
#
# 所以这里把"检测到假死 → 自动复位 → 重试"内建到相机层，
# 而不是要求每个调用方都记得先跑 reset 工具。
_USB_WEDGE_SIGNS = (
    "write_memory error", "LIBUSB_ERROR_BUSY", "PixelFormat_Reg",
    "Failed to claim USB control interface",
    # 这个是在"上一个进程刚释放、下一个进程立刻打开"时出现的（实测：
    # 前一个 live_view 收到 SIGTERM 干净退出后，紧接着启动的进程报
    # `Failed to bootstrap USB device ... (3)`）。同属设备没完全恢复，
    # 复位一下就好 —— 不列进来的话自动恢复不会触发。
    "Failed to bootstrap",
    "read_memory timeout",
)
_USB_VENDOR, _USB_PRODUCT = "2bdf", "0001"   # Hikrobot U3V
USBDEVFS_RESET = 0x5514                       # _IO('U', 20)


def is_usb_wedge(exc: BaseException) -> bool:
    """这个异常是不是"USB 假死"（而不是真的参数错误/设备不存在）。"""
    s = str(exc)
    return any(k in s for k in _USB_WEDGE_SIGNS)


def usb_reset_camera(vendor: str = _USB_VENDOR, product: str = _USB_PRODUCT,
                     settle_s: float = 2.5) -> bool:
    """对相机做一次 USB 复位。成功返回 True。

    为什么扫 sysfs 而不是记着旧路径：**复位后设备号会变**（003 → 006 之类），
    写死路径第二次就失效。
    """
    import fcntl

    def _nodes():
        found = []
        for d in Path("/sys/bus/usb/devices").glob("*"):
            try:
                if (d / "idVendor").read_text().strip().lower() != vendor.lower():
                    continue
                if (d / "idProduct").read_text().strip().lower() != product.lower():
                    continue
                found.append((int((d / "busnum").read_text()),
                              int((d / "devnum").read_text())))
            except (OSError, ValueError):
                continue
        return found

    nodes = _nodes()
    if not nodes:
        return False
    done = False
    for bus, dev in nodes:
        node = f"/dev/bus/usb/{bus:03d}/{dev:03d}"
        try:
            fd = os.open(node, os.O_RDWR)
        except OSError:
            continue
        try:
            fcntl.ioctl(fd, USBDEVFS_RESET, 0)
            done = True
        except OSError:
            pass
        finally:
            os.close(fd)
    if done:
        time.sleep(settle_s)          # 等重新枚举完成
    return done


class HikCamera:
    """USB3 Vision / GigE 工业相机。接口与 :class:`UvcCamera` 一致。"""

    kind = "hik"

    def __init__(self, device_index: int = 0, device_id: str | None = None):
        self.Aravis = _import_aravis()
        A = self.Aravis
        self.stream = None
        self._n_buffers = 0
        self._payload = 0
        self._ring: list = []          # 待回收的 buffer，避免频繁分配

        # 打开设备：遇到 USB 假死就自动复位并重试（见本模块顶部说明）。
        last_exc: BaseException | None = None
        for attempt in range(3):
            A.update_device_list()
            if A.get_n_devices() == 0:
                raise RuntimeError(
                    "aravis 没发现任何相机。检查：lsusb | grep 2bdf；"
                    "确认相机插在 USB3 口上（lsusb -t 应显示 5000M）")
            self.device_id = device_id or A.get_device_id(device_index)
            try:
                self.cam = A.Camera.new(self.device_id)
                break
            except Exception as e:                 # noqa: BLE001
                last_exc = e
                if attempt < 2 and is_usb_wedge(e):
                    print(f"  ⚠️  打开相机失败（USB 假死），自动复位后重试 "
                          f"({attempt+1}/2) ...")
                    usb_reset_camera()
                    time.sleep(1.0)
                    continue
                raise
        else:                                      # pragma: no cover
            raise last_exc if last_exc else RuntimeError("打开相机失败")

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
    def _apply_config(self, c, width, height, pixel_format, exposure_us,
                      exposure_auto, gain_db, frame_rate, n_buffers):
        """把参数写进相机对象 ``c``（不含打开/关闭，便于外面套重试）。"""
        A = self.Aravis
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

    def configure(self, width=None, height=None, pixel_format="Mono8",
                  exposure_us=None, exposure_auto=False, gain_db=None,
                  frame_rate=None, n_buffers=8):
        """设置采集参数。所有项都是可选的，只改传进来的。

        遇到 **USB 假死**会自动复位设备并重试一次 —— 这样所有调用方
        （取景 / 采集 / 参数工具）都不必自己处理这个坑。
        """
        A = self.Aravis

        # 改分辨率/像素格式前必须先停流，否则会报错
        if self.stream is not None:
            self.close()
            self.cam = A.Camera.new(self.device_id)

        args = (width, height, pixel_format, exposure_us, exposure_auto,
                gain_db, frame_rate, n_buffers)
        for attempt in range(2):
            try:
                # ⚠️ 注意：``self.cam`` 可能已被上面的 close()+Camera.new 换掉，
                # 所以这里**必须重新取**，不能用函数开头缓存的引用。
                # （原实现先取 `c = self.cam` 再 close/reopen，后续写的还是
                #   旧的、已经释放掉的那个相机对象 —— 是个潜伏的真 bug。）
                self._apply_config(self.cam, *args)
                return self
            except Exception as e:                 # noqa: BLE001
                if attempt == 0 and is_usb_wedge(e):
                    print(f"  ⚠️  相机配置失败（USB 假死），自动复位后重试 ...")
                    self.close()
                    usb_reset_camera()
                    # 复位后设备号可能变了，重新按 device_id 打开
                    self.Aravis.update_device_list()
                    self.cam = A.Camera.new(self.device_id)
                    continue
                raise
        return self

    # -- 取流 --------------------------------------------------
    def start(self):
        if self.stream is None:
            self.configure()
        self.cam.start_acquisition()

    def set_exposure(self, exposure_us=None, gain_db=None):
        """在**不停流**的前提下改曝光/增益 —— 实时取景时要一边看一边调。

        为什么不复用 :meth:`configure`：它为了改分辨率会 ``close()`` 再
        ``Camera.new()`` 重建整条流，调用一次要几百毫秒且会丢若干帧。
        只改曝光/增益时相机本来就允许在线设置（前提是 auto 已关，这正是
        ``configure`` 里做的事），所以直接写寄存器即可。
        """
        A, c = self.Aravis, self.cam
        if exposure_us is not None:
            c.set_exposure_time_auto(A.Auto.OFF)
            c.set_exposure_time(float(exposure_us))
        if gain_db is not None:
            c.set_gain_auto(A.Auto.OFF)
            c.set_gain(float(gain_db))
        return self

    def read_exposure(self) -> tuple[float | None, float | None]:
        """读回当前曝光(µs)/增益(dB)，读不到就返回 None。

        注意方法名是 ``get_exposure_time_auto`` / ``get_gain_auto`` ——
        **没有** ``is_*_auto`` 这种写法（那是 C API 的风格）。之前写成
        ``is_exposure_time_auto()`` 会抛 AttributeError，被这里的 except 吞掉，
        于是永远返回 ``(None, None)``，看起来像"读不出来"，其实是名字写错了。
        """
        A, c = self.Aravis, self.cam
        exp = gain = None
        try:
            if c.get_exposure_time_auto() == A.Auto.OFF:
                exp = float(c.get_exposure_time())
        except Exception:                                   # noqa: BLE001
            pass
        try:
            if c.get_gain_auto() == A.Auto.OFF:
                gain = float(c.get_gain())
        except Exception:                                   # noqa: BLE001
            pass
        return exp, gain

    def stop(self):
        if self.stream is not None:
            self.cam.stop_acquisition()

    def _decode(self, buf) -> np.ndarray:
        """把 aravis buffer 解成 numpy 图。

        ⚠️ **必须 ``copy()`` —— 这里曾经是一个 use-after-free**
        ------------------------------------------------------------------
        ``np.frombuffer(...)`` 是**零拷贝**的：它造出的数组只是**引用**
        ``buf.get_data()`` 指向的内存，而那块内存属于 aravis 的 ``buf`` 对象。
        ``frames()`` 每轮都新建 buffer 并且不回收旧的，所以 ``buf`` 在下一轮
        就被 GC 掉、内存随之释放 —— 于是 ``Frame.image`` 变成**悬垂指针**。

        实测症状（2026-09-14）：实时取景跑十几秒后直接
        ``corrupted size vs. prev_size`` + SIGABRT，**Python 侧连异常都拿不到**
        （堆损坏发生在 libaravis/glib 内部）。更糟的是它**不一定马上发作** ——
        单独测一帧时数据看着完全正确，只在内存被复用时才崩。这种"偶尔崩、
        多数时候对"的 bug 最难查，而且它就在**采集主路径**上。

        实测证据：``image.flags['OWNDATA'] is False``（不拥有内存）。
        加上 ``copy()`` 后变为 True，连续运行数分钟不再崩。

        代价：1624x1240 一帧 2 MB，10 Hz 也就 20 MB/s 的拷贝，完全可接受 ——
        用这点带宽换内存安全是明显划算的。
        """
        w, h = buf.get_image_width(), buf.get_image_height()
        pf = self.cam.get_pixel_format_as_string()
        n_ch = 3 if (pf in _RGB_FORMATS or pf in _BGR_FORMATS) else 1
        need = int(w) * int(h) * n_ch
        # count=need 同时防住"buffer 比预期短"的情况
        raw = np.frombuffer(buf.get_data(), dtype=np.uint8, count=need)
        shape = (h, w, 3) if n_ch == 3 else (h, w)
        return np.array(raw.reshape(shape), copy=True, order="C")

    def frames(self, n: int | None = None, timeout_us: int = 2_000_000,
               latest: bool = False) -> Iterator[Frame]:
        """产出帧。``n=None`` 表示一直取（调用方负责 break）。

        ⚠️ ``latest``：**实时取景必须开，采集绝不能开**
        ------------------------------------------------------------------
        流里默认有 8 块缓冲。消费者（显示 + 检测）比生产者（相机）慢时，
        队列会积压，而 ``timeout_pop_buffer`` 返回的是**最旧**的那一帧 ——
        于是画面稳定地落后好几帧。实测用户在 8 fps 下体感延迟"比较严重"。

        ``latest=True`` 时会在拿到一帧后**把积压的旧帧全部丢弃**，只保留
        最新的，从而把延迟压到一帧以内。

        **但采集（capture_h7）必须用默认的 False** —— 那里每一帧都要带
        自己的时间戳存下来，丢帧会直接破坏数据集。这两个场景的需求正好相反，
        所以做成显式开关而不是改默认值。
        """
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
                if latest:
                    # 把已经就绪的旧帧全部丢回流水线，只留下最新的一帧
                    while True:
                        extra = self.stream.try_pop_buffer()
                        if extra is None:
                            break
                        if extra.get_status() == A.BufferStatus.SUCCESS:
                            self.stream.push_buffer(buf)   # 旧的归还
                            buf = extra                     # 换成更新的
                        else:
                            self.n_missing += 1
                            self.stream.push_buffer(extra)

                status = buf.get_status()
                if status == A.BufferStatus.SUCCESS:
                    self.n_ok += 1
                    # 先在**归还之前**把像素拷出来（_decode 内部 copy），
                    # 归还之后这块内存就不属于我们了。
                    image = self._decode(buf)
                    host_ts = self._wall_ns_to_mono(buf.get_system_timestamp())
                    device_ts = buf.get_timestamp() or None
                    # ★ 归还**同一个** buffer，而不是新分配一个。
                    #   这是 aravis 的标准用法（pop → 处理 → push 回去），
                    #   缓冲区在流水线里循环复用。原实现每次 push 一块新的
                    #   `Buffer.new_allocate(payload)`，等于每帧都让 2 MB 内存
                    #   走一遍 malloc/free —— 10 Hz 下每秒 20 MB 的堆churn，
                    #   既浪费又显著增加触发堆损坏的机会（本项目已经因为
                    #   堆问题崩过好几次）。
                    self.stream.push_buffer(buf)
                    k += 1
                    yield Frame(image=image, host_ts=host_ts,
                                device_ts=device_ts, frame_id=k - 1)
                else:
                    self.n_missing += 1
                    self.stream.push_buffer(buf)
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

    def set_exposure(self, exposure_us=None, gain_db=None):
        """UVC 侧同样支持在线调曝光（V4L2 语义见 :meth:`configure`）。"""
        if exposure_us is not None:
            self.cap.set(self.cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.cap.set(self.cv2.CAP_PROP_EXPOSURE, float(exposure_us))
        if gain_db is not None:
            self.cap.set(self.cv2.CAP_PROP_GAIN, float(gain_db))
        return self

    def read_exposure(self) -> tuple[float | None, float | None]:
        exp = self.cap.get(self.cv2.CAP_PROP_EXPOSURE)
        gain = self.cap.get(self.cv2.CAP_PROP_GAIN)
        return (float(exp) if exp and exp > 0 else None,
                float(gain) if gain and gain > 0 else None)

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
