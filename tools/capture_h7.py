#!/usr/bin/env python3
"""采集 Camera-IMU 标定数据集（适配本机实测硬件）。

硬件组合（2026-09-13 实测通过）：
  · 相机：UVC 摄像头（笔记本 ASUS FHD webcam，``/dev/video0``）
          —— 海康 MV-CS020-10UC 是 U3V 工业相机，**不能用 OpenCV 打开**，
             见 doc/相机取流诊断.md；取流打通后换 ``--camera-source hik`` 即可。
  · IMU ：H7_IMU_With_EKF（VID:PID ``0483:6666``，``/dev/ttyACM0``，1 kHz）

产出（与 README 第 5 节声明格式一致）：:

    <out>/
    ├── images/000000.png ...      采集到的图像
    ├── image_timestamps.txt       "<文件名> <主机单调时钟秒>"
    ├── imu.txt                    "# timestamp gx gy gz ax ay az"（主机时钟，SI 单位）
    ├── imu_h7_full.csv            板载毫秒 + 六轴 + 欧拉角 + 四元数 + 主机时钟
    ├── imu_raw.bin                原始串口字节流（可事后重放解析，复核用）
    └── meta.json                  本次采集的参数与环境

用法::

    # 0) 只验 IMU，不接相机（最常用来确认硬件还活着）
    .venv/bin/python tools/capture_h7.py --imu-only --seconds 3 --out data/probe

    # 1) 无窗口自测：抓 3 帧，验证整条链路
    .venv/bin/python tools/capture_h7.py --out /tmp/selftest --frames 3

    # 2) 正式采集（交互，需要图形界面）
    .venv/bin/python tools/capture_h7.py --out data/session_01 \
        --camera 0 --width 640 --height 480 --serial /dev/ttyACM0

交互按键：``s`` 存一帧 / ``a`` 自动连拍开关 / ``q`` 结束

采集动作要求（决定标定精度的是动作，不是算法）：
  1. 开头保持 2 秒完全静止（零偏标定 + 重力对齐）；
  2. 绕 x / y / z **三个轴分别**缓慢转动若干圈（只绕一个轴 → 手眼标定无解）；
  3. 手持做**小幅平移**（纯旋转定不出平移外参）；
  4. 标定板走遍画面四角与中心，并覆盖不同倾角；
  5. 全程相机与 IMU 的**相对位置绝对不能变**。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

# 让脚本能直接 import calib/ 下的模块，而不依赖安装成包
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calib"))

from imu_h7 import H7Imu, validate  # noqa: E402

IMU_HEADER = "# timestamp gx gy gz ax ay az   (s, rad/s x3, m/s^2 x3; timestamp = 主机单调时钟)"
FULL_HEADER = (
    "t_board_ms,t_host_s,gx,gy,gz,ax,ay,az,roll,pitch,yaw,qw,qx,qy,qz"
)


def open_camera(index: int, width: int, height: int, lock_exposure: bool):
    """打开 UVC 相机。返回 cv2.VideoCapture，失败抛 RuntimeError。"""
    import cv2

    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"打不开相机 index={index}。若这是海康工业相机，请改用 MVS SDK 路线"
            f"（它不是 UVC 设备，OpenCV 无法打开）。"
        )
    # 关键：把驱动内部缓冲压到 1 帧。默认缓冲会让"读到帧的时刻"比"曝光时刻"
    # 晚几十毫秒且不稳定，这是 Camera-IMU 时间对齐最常见的精度杀手。
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if lock_exposure:
        # V4L2 语义：AUTO_EXPOSURE=1 手动、=3 自动。不同驱动实现有差异，
        # 失败也不致命，只提示。
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    got_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"相机已打开: index={index} 实际 {got_w}x{got_h} @ {got_fps:.1f} fps")
    if (got_w, got_h) != (width, height):
        print(f"  ⚠️  请求 {width}x{height}，实际拿到 {got_w}x{got_h}，"
              f"配置文件里的 image_size 要按**实际值**填")
    return cap


class ImuCollector:
    """后台把 IMU 帧全收进列表，供采集结束时一次性落盘。"""

    def __init__(self, imu: H7Imu):
        self.imu = imu
        self.frames = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            for f in self.imu.stream(timeout=0.3):
                self.frames.append(f)

    def stop(self):
        self._stop.set()
        self._t.join(timeout=1.0)


def write_outputs(out: Path, frames, imu_frames, raw: bytes, cam_info: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)

    if imu_frames:
        with (out / "imu.txt").open("w") as f:
            f.write(IMU_HEADER + "\n")
            for fr in imu_frames:
                g = fr.six_axis()          # gx gy gz ax ay az
                f.write(f"{fr.t_host:.6f} " + " ".join(f"{v:.6f}" for v in g) + "\n")

        with (out / "imu_h7_full.csv").open("w") as f:
            f.write(FULL_HEADER + "\n")
            for fr in imu_frames:
                v = fr.values
                f.write(
                    f"{fr.t_board_ms},{fr.t_host:.6f},"
                    f"{v['gx']:.6f},{v['gy']:.6f},{v['gz']:.6f},"
                    f"{v['ax']:.6f},{v['ay']:.6f},{v['az']:.6f},"
                    f"{v['roll']:.6f},{v['pitch']:.6f},{v['yaw']:.6f},"
                    f"{v['qw']:.6f},{v['qx']:.6f},{v['qy']:.6f},{v['qz']:.6f}\n"
                )

    if raw:
        (out / "imu_raw.bin").write_bytes(raw)

    meta = {
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_images": len(list((out / "images").glob("*.png"))) if (out / "images").exists() else 0,
        "n_imu_frames": len(imu_frames),
        "imu_raw_bytes": len(raw),
        **cam_info,
    }
    if imu_frames:
        meta["imu_check"] = validate(imu_frames)
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n已写入 {out}/")
    print(f"  图像      {meta['n_images']} 张")
    print(f"  IMU       {len(imu_frames)} 帧 ({len(raw)} 字节原始流)")
    if imu_frames:
        st = meta["imu_check"]
        print(f"  采样率    {st['采样率_Hz']:.3f} Hz")
        print(f"  时间戳步长非1的帧数  {st['时间戳步长_非1帧数']}  (应为 0)")
        print(f"  四元数vs欧拉角最大误差 {st['四元数vs欧拉角_最大误差_rad']:.2e} rad")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Camera-IMU 数据集采集（UVC 相机 + H7 IMU）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", required=True, help="输出目录，如 data/session_01")
    ap.add_argument("--serial", default="/dev/ttyACM0", help="IMU 串口")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--camera", default="0", help="相机索引，或 none")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--lock-exposure", action="store_true", help="尝试锁定自动曝光")
    ap.add_argument("--frames", type=int, default=0,
                    help=">0 则进入无窗口自测：自动抓 N 帧后退出")
    ap.add_argument("--frame-interval", type=float, default=0.3,
                    help="无窗口模式下每帧间隔秒数")
    ap.add_argument("--auto-interval", type=float, default=0.25,
                    help="交互模式下自动连拍的间隔秒数")
    ap.add_argument("--imu-only", action="store_true", help="只采 IMU，不需要相机")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="配合 --imu-only：采集时长（秒）")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    (out / "images").mkdir(parents=True, exist_ok=True)

    # ── IMU ────────────────────────────────────────────────
    print(f"打开 IMU {args.serial} @ {args.baud} ...")
    try:
        imu = H7Imu(args.serial, args.baud, keep_raw=True)
    except Exception as e:  # noqa: BLE001
        print(f"❌ 打不开串口：{e}")
        print("   若提示权限不足：把用户加进 dialout 组并重新登录，")
        print("   或执行 sudo bash tools/setup_permissions.sh 写入 udev 规则。")
        return 2
    imu.start()
    collector = ImuCollector(imu)
    collector.start()

    # 等第一批帧，确认协议对得上
    time.sleep(0.4)
    if imu.n_frames == 0:
        print("❌ 串口打开成功但收不到数据。确认：设备是否在发送？波特率/线是否正确？")
        print("   排查工具：python3 calib/imu_h7.py --port <串口>")
        collector.stop(); imu.close()
        return 2
    print(f"IMU 已在收数据（{imu.n_frames} 帧 / {imu.n_bytes} 字节）")

    # ── 只采 IMU ───────────────────────────────────────────
    if args.imu_only:
        secs = args.seconds if args.seconds > 0 else 5.0
        print(f"IMU-only 模式，采集 {secs} 秒 ...")
        time.sleep(secs)
        collector.stop(); imu.close()
        write_outputs(out, [], collector.frames, bytes(imu.raw), {"camera": "none"})
        return 0

    # ── 相机 ───────────────────────────────────────────────
    cam_info = {"camera": args.camera}
    cap = None
    if args.camera.lower() != "none":
        try:
            cap = open_camera(int(args.camera), args.width, args.height,
                              args.lock_exposure)
            cam_info.update(
                camera_width=int(cap.get(3)), camera_height=int(cap.get(4)),
                camera_fps=float(cap.get(5)),
            )
        except Exception as e:  # noqa: BLE001
            print(f"❌ {e}")
            print("   可以先用 --camera none 只采 IMU，或先用笔记本摄像头跑通流程。")
            collector.stop(); imu.close()
            return 2

    ts_lines: list[str] = []
    idx = 0

    def save_frame(frame) -> None:
        nonlocal idx
        import cv2
        t = time.monotonic()          # 紧跟 read() 之后取时刻，越早越准
        name = f"{idx:06d}.png"
        cv2.imwrite(str(out / "images" / name), frame)
        ts_lines.append(f"{name} {t:.9f}")
        idx += 1

    try:
        if args.frames > 0:
            # 无窗口自测
            print(f"无窗口模式：抓 {args.frames} 帧 ...")
            for _ in range(args.frames):
                ok, frame = cap.read()
                if not ok:
                    print("❌ 读帧失败")
                    break
                save_frame(frame)
                time.sleep(args.frame_interval)
        else:
            import cv2
            print("\n交互模式：s=存一帧  a=自动连拍  q=结束")
            auto = False
            last_auto = 0.0
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("❌ 读帧失败，退出")
                    break
                now = time.monotonic()
                if auto and now - last_auto >= args.auto_interval:
                    save_frame(frame)
                    last_auto = now
                disp = frame.copy()
                cv2.putText(disp, f"saved={idx} auto={'ON' if auto else 'off'}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if auto else (0, 200, 255), 2)
                cv2.imshow("capture (s save / a auto / q quit)", disp)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("s"):
                    save_frame(frame)
                elif k == ord("a"):
                    auto = not auto
                    print(f"自动连拍 {'开' if auto else '关'}")
                elif k == ord("q"):
                    break
            cv2.destroyAllWindows()
    finally:
        if cap is not None:
            cap.release()
        collector.stop()
        imu.close()

    with (out / "image_timestamps.txt").open("w") as f:
        f.write("# filename timestamp_seconds (monotonic clock, unit: s)\n")
        f.write("\n".join(ts_lines) + ("\n" if ts_lines else ""))

    write_outputs(out, [], collector.frames, bytes(imu.raw), cam_info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
