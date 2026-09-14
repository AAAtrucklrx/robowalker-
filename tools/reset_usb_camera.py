#!/usr/bin/env python3
"""把卡住的工业相机从"USB 假死"里救回来（**不需要 root**）。

⚠️ 先看这条：**相机层现在会自愈，本工具通常不用手动跑了**
==================================================================
``calib/camera.py`` 已内建"检测到假死 → 自动 USB 复位 → 重试"，
``live_view.py`` / ``capture_h7.py`` / ``hunt_board.py`` 遇到假死会自己恢复，
不会再出现"一打开就是黑屏"。本工具保留作**兜底**：自愈失败、想在不打开
相机的前提下确认设备状态、或排查"硬件还是软件"时用。

为什么需要这个工具
==================================================================
2026-09-14 实测踩到：只要有一次**不干净释放**（进程被强杀 / SIGABRT /
忘了 ``close()``），相机就进入假死，症状**逐步恶化**且每步报错都不一样，
很容易误判成"相机坏了"或"代码错了"：

1. 先是 ``corrupted size vs. prev_size`` —— 直接 SIGABRT（glibc 堆损坏，在
   libaravis 内部），Python 侧连异常都拿不到；
2. 再跑变成 ``[PixelFormat_Reg] USB3Vision write_memory error``
   —— 看起来像"这个像素格式不支持"，其实是设备不响应 USB 控制写了；
3. 之后每次都 ``Failed to claim USB control interface ... LIBUSB_ERROR_BUSY``
   —— 看起来像"有别的进程占着"，其实没有任何进程占着，是设备自己没释放。

这三条报错**没有一条**指向真正的原因。

**定因实验**：5 次「打开→配置→取帧→正常 close()」全部成功；
而只要不干净释放，第 2 次就复现 write_memory error。
⇒ **假死 = 没有干净释放**，不是相机硬件故障。

恢复办法是对 USB 设备做一次 ``USBDEVFS_RESET`` ioctl。实测本机
``/dev/bus/usb/004/003`` 是 ``crw-rw-rw-``，所以**不用 sudo**。

注意：复位后**设备号会变**（003 → 006 之类），所以必须重新扫描 sysfs，
不能记着旧路径。

用法::

    .venv/bin/python tools/reset_usb_camera.py            # 复位 + 验证能取流
    .venv/bin/python tools/reset_usb_camera.py --no-verify
"""
from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calib"))

USBDEVFS_RESET = 0x5514          # _IO('U', 20)
VENDOR, PRODUCT = "2bdf", "0001"  # Hikrobot MV-CS020-10UC


def find_usb_nodes(vendor: str = VENDOR, product: str = PRODUCT) -> list[tuple[str, str]]:
    """在 sysfs 里找匹配 vid:pid 的 USB 设备，返回 [(busnum, devnum), ...]。

    为什么不解析 `lsusb` 的输出：它的格式随版本变，而且读到 bus/dev 号之后
    还要知道**当前**的号（复位后会变）。sysfs 是权威来源，直接读。
    """
    out = []
    base = Path("/sys/bus/usb/devices")
    for d in base.glob("*"):
        try:
            v = (d / "idVendor").read_text().strip()
            p = (d / "idProduct").read_text().strip()
            if v.lower() == vendor.lower() and p.lower() == product.lower():
                bus = (d / "busnum").read_text().strip()
                dev = (d / "devnum").read_text().strip()
                out.append((bus, dev))
        except (OSError, ValueError):
            continue
    return out


def _usb_reset(bus: str, dev: str) -> bool:
    node = f"/dev/bus/usb/{int(bus):03d}/{int(dev):03d}"
    try:
        fd = os.open(node, os.O_RDWR)
    except OSError as e:
        print(f"  ❌ 打不开 {node}: {e}")
        return False
    try:
        fcntl.ioctl(fd, USBDEVFS_RESET, 0)
        print(f"  ✅ 已对 {node} 下发 USB 复位")
        return True
    except OSError as e:
        print(f"  ❌ {node} 复位失败: {e}")
        return False
    finally:
        os.close(fd)


def main() -> int:
    ap = argparse.ArgumentParser(description="USB 复位卡住的工业相机")
    ap.add_argument("--no-verify", action="store_true", help="只复位，不验证取流")
    ap.add_argument("--wait", type=float, default=15.0,
                    help="复位后等待重新枚举的秒数上限")
    args = ap.parse_args()

    print("=" * 66)
    print("工业相机 USB 复位")
    print("=" * 66)

    nodes = find_usb_nodes()
    if not nodes:
        print(f"❌ sysfs 里找不到 {VENDOR}:{PRODUCT} 的相机。")
        print("   请检查：① USB 线插好（USB3 口）；② 相机已上电。")
        print("   若设备仍在 lsusb 里但 sysfs 读不到，试试物理拔插。")
        return 1
    print(f"找到 {len(nodes)} 个匹配设备: " +
          ", ".join(f"bus {b} dev {d}" for b, d in nodes))

    ok = False
    for bus, dev in nodes:
        ok = _usb_reset(bus, dev) or ok
    if not ok:
        print("\n❌ 复位都没成功。物理拔插一次相机即可（复位是软件层面的努力）。")
        return 1

    print(f"\n等待重新枚举（最多 {args.wait:.0f} s）...")
    t0 = time.time()
    new_nodes: list[tuple[str, str]] = []
    while time.time() - t0 < args.wait:
        time.sleep(0.5)
        new_nodes = find_usb_nodes()
        if new_nodes:
            break
    if not new_nodes:
        print("❌ 复位后设备没有回来。请物理拔插。")
        return 1
    print(f"  ✅ 设备已重新枚举: " +
          ", ".join(f"bus {b} dev {d}" for b, d in new_nodes))

    if args.no_verify:
        return 0

    print("\n验证能否取流 ...")
    try:
        from camera import open_camera
        from exposure import exposure_stats
    except Exception as e:                       # noqa: BLE001
        print(f"  ⚠️  导入相机层失败（{e}），跳过验证")
        return 0

    for attempt in range(3):
        try:
            cap = open_camera("hik")
            # 只做一次最小配置，别连续快速开关 —— 那正是当初把设备搞挂的操作
            cap.configure(width=1624, height=1240, pixel_format="Mono8",
                          exposure_us=120000, gain_db=10, frame_rate=10)
            it = cap.frames(None)
            next(it)
            f = next(it)
            st = exposure_stats(f.image)
            print(f"  ✅ 取流正常 {f.image.shape}  mean={st['mean']:.1f} "
                  f"（曝光判定 {st['verdict']}）")
            cap.close()
            print("\n相机已恢复，可以继续用 tools/live_view.py 或 capture_h7.py。")
            return 0
        except Exception as e:                   # noqa: BLE001
            print(f"  第 {attempt+1} 次失败: {type(e).__name__}: {e}")
            time.sleep(3)

    print("\n⚠️  复位成功但取流仍失败。请物理拔插相机（并确认插在 USB3 口）。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
