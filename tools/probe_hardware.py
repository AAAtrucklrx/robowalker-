#!/usr/bin/env python3
"""硬件连接自检：一次性排查相机、串口 IMU 的可用性与权限。

用法:
    .venv/bin/python tools/probe_hardware.py            # 全部检查
    .venv/bin/python tools/probe_hardware.py --camera   # 只查相机
    .venv/bin/python tools/probe_hardware.py --serial   # 只查串口

设计原则：把"设备没连"、"权限不够"、"设备连了但读不出数据"三种失败区分开，
因为它们的修法完全不同。
"""
from __future__ import annotations

import argparse
import glob
import os
import pwd
import subprocess
import sys

OK, WARN, BAD = "✅", "⚠️ ", "❌"


def line(mark: str, text: str) -> None:
    print(f"{mark} {text}")


def check_python_deps() -> bool:
    print("\n=== 1. Python 依赖 ===")
    good = True
    for mod in ("numpy", "scipy", "cv2", "yaml"):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            line(OK, f"{mod:<10} {ver}")
        except Exception as exc:  # noqa: BLE001
            line(BAD, f"{mod:<10} 缺失: {exc}")
            good = False
    try:
        import cv2

        line(OK if hasattr(cv2, "aruco") else BAD, f"cv2.aruco 可用: {hasattr(cv2, 'aruco')}")
    except Exception:  # noqa: BLE001
        good = False
    return good


def check_camera(max_index: int = 8) -> bool:
    print("\n=== 2. 相机 ===")
    import cv2
    import numpy as np

    nodes = sorted(glob.glob("/dev/video*"))
    if not nodes:
        line(BAD, "系统里没有 /dev/video* 节点：相机未接入或被内核驱动占用")
        return False
    line(OK, f"发现视频节点: {', '.join(nodes)}")

    me = pwd.getpwuid(os.getuid()).pw_name
    any_usable = False
    for idx in range(max_index):
        node = f"/dev/video{idx}"
        if not os.path.exists(node):
            continue
        readable = os.access(node, os.R_OK | os.W_OK)
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            why = "权限不足" if not readable else "打不开（可能是元数据/子设备节点，正常）"
            line(WARN, f"{node}: {why}")
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            line(WARN, f"{node}: 打开成功 {w}x{h} 但读不到帧")
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
        verdict = "真实画面" if sharp > 50 and g.mean() > 5 else "疑似全黑/无内容（红外或快门关闭的那路）"
        line(OK, f"{node}: {w}x{h}@{fps:.0f}fps 亮度={g.mean():.1f} 清晰度={sharp:.0f} → {verdict}")
        if verdict == "真实画面":
            any_usable = True

    if not any_usable:
        line(BAD, "没有任何一路相机读出有效画面")
    print(f"   当前用户: {me}（/dev/video* 需要 video 组或 ACL 授权）")
    return any_usable


def check_serial() -> bool:
    print("\n=== 3. 串口 IMU ===")
    import serial  # 可能未安装

    # 只看真正由 USB 转串口芯片产生、以及内核 8250 驱动真的注册成功的口。
    # 主板上那一堆未接线的 legacy ttyS0-ttyS31 是噪音，不算失败。
    usb_nodes = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    real_s = []
    for n in sorted(glob.glob("/dev/ttyS[0-9]*")):
        try:
            dev = os.path.realpath(f"/sys/class/tty/{os.path.basename(n)}/device")
        except OSError:
            continue
        # 内核会给主板上的 legacy 口预先注册 serial8250 平台占位设备，
        # 它们没有真实 UART 时钟（/proc/tty/driver/serial 里没有对应行）。
        # 只有挂在真实总线（pci/平台 UART 控制器）下的口才算数。
        if "serial8250" in dev:
            continue
        real_s.append(n)
    nodes = usb_nodes + real_s
    legacy = len(glob.glob("/dev/ttyS[0-9]*")) - len(real_s)
    if legacy:
        line(OK, f"忽略 {legacy} 个 legacy ttyS* 占位口（serial8250 平台设备，无真实硬件）")
    if not nodes:
        line(WARN, "没有发现可用串口：IMU 还没插，或没被识别成串口")
        print("   插上后先跑: lsusb   和   dmesg | tail -20   看内核有没有认出来")
        print("   认出来的话会出现 /dev/ttyUSB0（CH340/CP210x 等）或 /dev/ttyACM0（原生 USB CDC）")
        print("   也可以用 /dev/serial/by-id/ 里的稳定名字，避免插拔后编号变化")
        return False

    groups = [g.gr_name for g in __import__("grp").getgrall() if pwd.getpwuid(os.getuid()).pw_name in g.gr_mem]
    in_dialout = "dialout" in groups
    usable = False
    for node in nodes:
        real = os.path.realpath(node)
        base = os.path.basename(real) if real.startswith("/dev/serial/by-id") else os.path.basename(node)
        st = os.stat(node)
        owner = pwd.getpwuid(st.st_uid).pw_name
        group = __import__("grp").getgrgid(st.st_gid).gr_name
        rw = os.access(node, os.R_OK | os.W_OK)
        mark = OK if rw else BAD
        line(mark, f"{node} → {base}  属主={owner}:{group}  当前用户{'可读写' if rw else '不可读写'}")
        usable = usable or rw
        if not rw:
            print(f"   修法之一: sudo usermod -aG dialout $USER  （需重新登录生效，且需要 sudo 密码）")
            print(f"   临时验证可用: sudo chmod a+rw {node}")
    if not in_dialout:
        line(WARN, "当前用户不在 dialout 组：插上新串口设备时大概率没有读写权限")
        if not usable:
            print("   → 结论：设备存在但用不了，先解决权限再采集")
    return usable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", action="store_true", help="只检查相机")
    ap.add_argument("--serial", action="store_true", help="只检查串口")
    args = ap.parse_args()
    only = args.camera or args.serial

    results = []
    if not only or args.camera:
        results.append(("相机", check_camera()))
    if not only or args.serial:
        try:
            results.append(("串口", check_serial()))
        except ImportError:
            print("\n=== 3. 串口 IMU ===")
            line(WARN, "pyserial 未安装: .venv/bin/python -m pip install pyserial")
            results.append(("串口", False))

    print("\n=== 结论 ===")
    for name, ok in results:
        line(OK if ok else WARN, f"{name}: {'可用' if ok else '尚不可用'}")
    print("\n提示: 装完依赖后先跑本脚本，再跑 tools/capture_dataset.py 采数据。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
