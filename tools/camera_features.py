#!/usr/bin/env python3
"""把相机的 GenICam 参数**过一遍**：挨个探测标准参数名，打印存在项与当前值。

为什么需要这个工具
==================================================================
aravis 的 Python 绑定**没有暴露 feature 枚举接口**（``Camera`` 上只有
``is_feature_available`` / ``is_feature_implemented``，没有 ``get_feature_names``；
顶层也没有）。官方的 ``arv-tool-0.8`` 又**不在 ``aravis-tools`` 包里**，
``arv-camera-test --features list`` 也不认 list。所以想"看看这台相机到底有哪些
参数、现在是什么值"，唯一方便的路子就是**按 GenICam SFNC 标准名逐个探测**。

本工具就是这么做的：列一份标准名清单，挨个用
``is_feature_available`` + 对应的 ``get_*_feature`` 读一遍，分成几组打印，
并把**对本次标定真正有影响的**项标出来。

用法::

    .venv/bin/python tools/camera_features.py
    .venv/bin/python tools/camera_features.py --stream        # 顺带测一下取流
    .venv/bin/python tools/camera_features.py --set ExposureTime=80000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calib"))


# 分组：(组名, [(feature 名, 类型, 备注)], ...)
# 类型 s/i/f/b 对应 aravis 的 string/integer/float/boolean
GROUPS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("设备信息", [
        ("DeviceVendorName", "s", ""),
        ("DeviceModelName", "s", ""),
        ("DeviceVersion", "s", ""),
        ("DeviceFirmwareVersion", "s", ""),
        ("DeviceSerialNumber", "s", ""),
        ("DeviceID", "s", ""),
        ("DeviceUserID", "s", "可自定义的设备名"),
        ("DeviceManufacturerInfo", "s", ""),
    ]),
    ("图像格式 / 几何", [
        ("Width", "i", "★标定分辨率"),
        ("Height", "i", "★标定分辨率"),
        ("WidthMax", "i", ""),
        ("HeightMax", "i", ""),
        ("PixelFormat", "s", "★必须 Mono8"),
        ("PixelSize", "f", ""),
        ("BinningHorizontal", "i", "开了会改变内参"),
        ("BinningVertical", "i", "开了会改变内参"),
        ("DecimationHorizontal", "i", ""),
        ("DecimationVertical", "i", ""),
        ("ReverseX", "b", "会改变像素坐标方向"),
        ("ReverseY", "b", "会改变像素坐标方向"),
    ]),
    ("曝光 / 增益 ★采集核心", [
        ("ExposureTime", "f", "★曝光 µs"),
        ("ExposureTimeAbs", "f", "老式命名"),
        ("ExposureAuto", "s", "★必须是 Off"),
        ("ExposureMode", "s", ""),
        ("Gain", "f", "★增益 dB"),
        ("GainRaw", "i", "老式命名"),
        ("GainAuto", "s", "★必须是 Off"),
        ("GainSelector", "s", ""),
        ("BlackLevel", "f", "影响黑场"),
        ("Gamma", "f", "非线性，标定前要关"),
        ("GammaEnable", "b", "★应关掉"),
        ("DigitalShift", "i", ""),
    ]),
    ("采集控制", [
        ("AcquisitionMode", "s", "★Continuous"),
        ("AcquisitionFrameRate", "f", "★帧率"),
        ("AcquisitionFrameRateEnable", "b", "★不开则跑满"),
        ("AcquisitionFrameRateAbs", "f", "老式命名"),
        ("AcquisitionFrameCount", "i", ""),
        ("AcquisitionBurstFrameCount", "i", ""),
        ("TriggerMode", "s", "★必须 Off（自由跑）"),
        ("TriggerSource", "s", ""),
        ("TriggerActivation", "s", ""),
        ("TriggerSelector", "s", ""),
        ("TriggerSoftware", "c", "命令，不可读"),
    ]),
    ("镜头 / 对焦（有则可用）", [
        ("FocusPos", "i", "★电动对焦位置"),
        ("FocusAbs", "f", ""),
        ("FocusAuto", "s", ""),
        ("FocusMin", "i", ""),
        ("FocusMax", "i", ""),
        ("FocusStep", "i", ""),
    ]),
    ("传输 / 带宽（USB3 相关）", [
        ("DeviceLinkSpeed", "i", "应 5000000000 (USB3 5Gbps)"),
        ("DeviceLinkThroughputLimit", "i", "★限速，影响丢帧"),
        ("DeviceLinkThroughputLimitMode", "s", ""),
        ("PayloadSize", "i", ""),
        ("TimestampTickFrequency", "i", "★硬件时间戳 tick"),
        ("GevTimestampTickFrequency", "i", "GigE 版命名"),
        ("DeviceLinkHeartbeatTimeout", "f", ""),
    ]),
    ("其他可能存在的", [
        ("ChunkModeActive", "b", ""),
        ("ChunkSelector", "s", ""),
        ("LineSelector", "s", ""),
        ("LineMode", "s", ""),
        ("LineSource", "s", ""),
        ("LineInverter", "b", ""),
        ("UserSetSelector", "s", ""),
        ("UserSetDefault", "s", ""),
        ("BalanceWhiteAuto", "s", ""),
        ("LUTEnable", "b", ""),
        ("ContrastEnable", "b", ""),
        ("SharpnessEnable", "b", ""),
    ]),
]

_GETTERS = {"s": "get_string", "i": "get_integer",
            "f": "get_float", "b": "get_boolean"}
# 注意：aravis 的 Python 绑定用的是 ``get_integer(name)`` 这种**不带 feature 后缀**
# 的名字，而不是 C API 的 ``arv_camera_get_integer_feature``。写成
# ``get_integer_feature`` 会 ``is_feature_available`` 返回 True、取值时却
# AttributeError —— 看起来像"参数不存在"，其实只是方法名写错了。
_SETTERS = {"s": "set_string", "i": "set_integer",
            "f": "set_float", "b": "set_boolean"}


def dump(cam, verbose: bool = True) -> dict:
    """按分组探测并打印；返回 {feature: value} 供程序使用。"""
    found: dict = {}
    missing: list[str] = []
    for gname, items in GROUPS:
        rows = []
        for name, typ, note in items:
            if typ == "c":
                continue                      # 命令型，跳过
            try:
                if not cam.is_feature_available(name):
                    missing.append(name)
                    continue
                val = getattr(cam, _GETTERS[typ])(name)
            except Exception as e:             # noqa: BLE001
                missing.append(name)
                if verbose:
                    rows.append((name, f"<读失败 {type(e).__name__}>", note))
                continue
            found[name] = val
            rows.append((name, val, note))
        if rows:
            print(f"\n【{gname}】")
            for name, val, note in rows:
                vs = f"{val:.6g}" if isinstance(val, float) else str(val)
                mark = "  " if not note else ""
                print(f"  {name:<28} = {vs:<24}{mark}{note}")
        elif verbose:
            print(f"\n【{gname}】 (无可用参数)")
    return {"found": found, "missing": missing}


def main() -> int:
    ap = argparse.ArgumentParser(description="过一遍相机的 GenICam 参数")
    ap.add_argument("--set", action="append", default=[],
                    metavar="NAME=VALUE", help="写参数（可多次）")
    ap.add_argument("--stream", action="store_true", help="顺带取 3 帧验证")
    ap.add_argument("--exposure", type=float, default=120000.0)
    ap.add_argument("--gain", type=float, default=10.0)
    args = ap.parse_args()

    import gi
    gi.require_version("Aravis", "0.8")
    from gi.repository import Aravis as A

    A.update_device_list()
    if A.get_n_devices() == 0:
        print("❌ aravis 没发现相机。检查 lsusb | grep 2bdf，"
              "或先跑 tools/reset_usb_camera.py")
        return 1
    dev_id = A.get_device_id(0)
    print("=" * 72)
    print(f"相机参数总览  ——  {dev_id}")
    print("=" * 72)

    cam = A.Camera.new(dev_id)
    res = dump(cam)

    if res["missing"]:
        print(f"\n（另有 {len(res['missing'])} 个标准名在本机不可用，"
              f"未列出：{', '.join(res['missing'][:8])}"
              f"{' …' if len(res['missing']) > 8 else ''}）")

    # 写参数
    for item in args.set:
        if "=" not in item:
            print(f"⚠️  忽略无法解析的 --set {item!r}（需要 NAME=VALUE）")
            continue
        name, raw = item.split("=", 1)
        name = name.strip()
        for typ in ("i", "f", "s", "b"):
            getter, setter = _GETTERS[typ], _SETTERS[typ]
            try:
                if not cam.is_feature_available(name):
                    continue
                old = getattr(cam, getter)(name)
                val = (raw.lower() in ("1", "true", "on")) if typ == "b" else (
                    raw if typ == "s" else (int(raw) if typ == "i" else float(raw)))
                getattr(cam, setter)(name, val)
                new = getattr(cam, getter)(name)
                print(f"\n✏️  {name}: {old} → {new}")
                break
            except Exception as e:              # noqa: BLE001
                print(f"\n❌ 写 {name} 失败: {type(e).__name__}: {e}")
                break

    # 标定相关的几条硬性检查
    print("\n" + "=" * 72)
    print("对本项目（Camera-IMU 标定）的检查")
    print("=" * 72)
    f = res["found"]
    checks = [
        ("PixelFormat", lambda v: str(v).startswith("Mono"),
         "必须是 Mono8（灰度，角点检测与内参模型都基于单通道）"),
        ("ExposureAuto", lambda v: str(v).lower() == "off",
         "必须 Off —— 自动曝光会让不同帧亮度不一致且随时变"),
        ("GainAuto", lambda v: str(v).lower() == "off",
         "必须 Off —— 自动增益同理"),
        ("TriggerMode", lambda v: str(v).lower() == "off",
         "必须 Off —— 自由跑才和 IMU 时间戳可比"),
        ("GammaEnable", lambda v: not v, "建议关掉 —— 非线性会破坏针孔+畸变模型"),
        ("ReverseX", lambda v: not v, "建议 Off —— 翻转会改变像素坐标手性"),
        ("ReverseY", lambda v: not v, "同上"),
        ("AcquisitionFrameRateEnable", lambda v: bool(v),
         "建议开 —— 固定帧率，帧间隔才稳定可预测"),
    ]
    for name, ok_fn, why in checks:
        if name not in f:
            continue
        v = f[name]
        try:
            good = ok_fn(v)
        except Exception:                       # noqa: BLE001
            good = None
        mark = "✅" if good else ("⚠️ " if good is False else "? ")
        print(f"  {mark} {name} = {v}   {why}")

    # 取流验证
    if args.stream:
        print("\n" + "=" * 72)
        print("取流验证")
        print("=" * 72)
        from exposure import exposure_stats
        for name, val in (("ExposureTime", args.exposure), ("Gain", args.gain)):
            try:
                if cam.is_feature_available(name):
                    if name == "ExposureTime":
                        cam.set_exposure_time_auto(A.Auto.OFF)
                        cam.set_exposure_time(val)
                    else:
                        cam.set_gain_auto(A.Auto.OFF)
                        cam.set_gain(val)
            except Exception as e:              # noqa: BLE001
                print(f"  ⚠️  设置 {name} 失败: {e}")
        w = f.get("Width", 1624); h = f.get("Height", 1240)
        cam.set_integer("Width", int(w)); cam.set_integer("Height", int(h))
        payload = cam.get_payload()
        stream = cam.create_stream(None, None, 8)
        for _ in range(8):
            stream.push_buffer(A.Buffer.new_allocate(payload))
        cam.start_acquisition()
        try:
            for k in range(3):
                buf = stream.timeout_pop_buffer(2_000_000)
                if buf is None or buf.get_status() != A.BufferStatus.SUCCESS:
                    print(f"  第{k+1}帧失败")
                    continue
                import numpy as np
                g = np.frombuffer(buf.get_data(), np.uint8,
                                  count=int(w) * int(h)).reshape(int(h), int(w))
                st = exposure_stats(g)
                print(f"  第{k+1}帧 {g.shape} mean={st['mean']:.1f} "
                      f"饱和={st['sat_hi']*100:.1f}% → {st['verdict']}")
        finally:
            cam.stop_acquisition()

    cam = None
    import gc; gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
