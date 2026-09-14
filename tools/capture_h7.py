#!/usr/bin/env python3
"""采集 Camera-IMU 标定数据集（统一相机层：海康 U3V 工业相机 / 普通 UVC）。

硬件组合
------------------------------------------------------------------
* **海康 MV-CS020-10UC**（USB3 Vision）→ ``--camera-source hik``
  1624×1240 Mono8，支持 ``DeviceTimestamp`` 硬件时间戳
* **UVC 摄像头**（笔记本自带）→ ``--camera-source 0``
  算法开发阶段的默认数据源
* **H7_IMU_With_EKF**（``0483:6666``）→ ``--serial /dev/ttyACM0``，1 kHz

采集质量反馈（这一版新增，来自实测结论）
------------------------------------------------------------------
C2 时间对齐的精度满足一条硬约束：

    **τ 的时间分辨率 ≈ 角度残差 / 角速度**

实测：合成轨迹只有 31.6 °/s 时，角度残差 0.24° 直接等价成 **7.6 ms** 的
时间不确定度。所以**转得慢 = 时间对齐必不准**。

本工具会在采集时实时显示角速度，并在太慢时给出红色警告。
目标：**≥ 90 °/s**。

产出（与 README 第 5 节格式一致）::

    <out>/
    ├── images/000000.png ...
    ├── image_timestamps.txt     "<文件名> <主机单调时钟秒>"  ← 标定程序读这个
    ├── camera_timestamps.csv    相机时间戳全量（含设备硬件时间戳，若有）
    ├── imu.txt                  "# timestamp gx gy gz ax ay az"（主机时钟，SI）
    ├── imu_h7_full.csv          板载毫秒 + 六轴 + 欧拉角 + 四元数 + 主机时钟
    ├── imu_raw.bin              原始串口字节流（可事后重放复核）
    └── meta.json                采集参数与环境

用法::

    # 0) 只验 IMU
    .venv/bin/python tools/capture_h7.py --imu-only --seconds 3 --out data/probe

    # 1) 无窗口自测
    .venv/bin/python tools/capture_h7.py --out /tmp/selftest --frames 3

    # 2) 工业相机正式采集
    .venv/bin/python tools/capture_h7.py --out data/session_01 \\
        --camera-source hik --width 1624 --height 1240 --pixel-format Mono8 \\
        --exposure 20000 --gain 12 --frame-rate 10 --serial /dev/ttyACM0

    # 3) UVC 相机
    .venv/bin/python tools/capture_h7.py --out data/session_01 \\
        --camera-source 0 --width 640 --height 480

交互按键：``s`` 存一帧 / ``a`` 自动连拍 / ``q`` 结束

**采集动作要求（决定标定精度的是动作，不是算法）**

1. 开头 **静置 3 秒**（零偏标定 + 重力对齐**只能**靠静止段，这段没有就全废）；
2. 绕 x / y / z **三个轴分别**快速转动若干圈 —— **手腕要快，目标 ≥ 90 °/s**；
   只绕一个轴转 → 手眼标定在数学上退化；
3. 做**小幅平移**（纯旋转定不出平移外参）；
4. 标定板走遍画面四角与中心，覆盖不同倾角；
5. 全程相机与 IMU 的**相对位置绝对不能变**（USB 线别拽着 IMU）。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calib"))

from camera import open_camera  # noqa: E402
from exposure import exposure_stats  # noqa: E402
from imu_h7 import H7Imu, validate  # noqa: E402
from target_detect import BoardSpec  # noqa: E402

IMU_HEADER = ("# timestamp gx gy gz ax ay az   "
              "(s, rad/s x3, m/s^2 x3; timestamp = 主机单调时钟)")
FULL_HEADER = "t_board_ms,t_host_s,gx,gy,gz,ax,ay,az,roll,pitch,yaw,qw,qx,qy,qz"

# 目标角速度（来自实测：τ 分辨率 ≈ 角度残差 / 角速度）
TARGET_GYRO_DPS = 90.0
MIN_GYRO_DPS = 40.0


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

    def recent_gyro_dps(self, n: int = 200) -> float:
        """最近 n 帧的平均角速度模长（度/秒）—— 采集质量实时反馈。"""
        import numpy as np

        if len(self.frames) < 10:
            return 0.0
        g = np.array([f.gyro for f in self.frames[-n:]])
        return float(np.rad2deg(np.linalg.norm(g, axis=1).mean()))


class BoardTracker:
    """用标定板的**表观尺寸**估计深度跨度 —— 采集时的硬性质量指标。

    为什么需要它
    ------------------------------------------------------------------
    受控实验证明（``tools/experiment_depth_diversity.py``）：相机相对标定板的
    **深度跨度**决定内参能不能解出来。

    | 深度跨度 | fx 误差 | k3（真值 0） | 重投影 RMS |
    |---|---|---|---|
    | 1.0x | **+4.17%** | -0.047 | 0.128 px |
    | 1.8x | -0.12% | -0.013 | 0.149 px |
    | 3.3x | +0.31% | +0.004 | 0.153 px |

    **三行的重投影误差几乎一样** —— 只在同一距离换角度，内参是错的但残差不报警。

    原理：板在画面里的面积 ∝ 1/z²，所以 ``sqrt(area)`` 的 max/min **就是深度跨度**。
    不需要相机内参，采集时就能实时判断。

    实现上为了不拖慢采集，每 ``every`` 帧才检测一次，且先降采样到 ~640 宽。

    离线验证（喂 synth_01 的 180 帧）：估计 2.15x，轨迹真值 1.92x，
    偏差 12% 且偏保守方向 —— 作为采集时的实时指标足够。
    （它用 5/95 分位而不是极值，所以会略偏小/偏稳；投影面积还受板姿态影响，
    所以这是**代理指标**，用来发现"深度基本没变"这种致命情况。）
    """

    def __init__(self, spec, every: int = 5, downscale_to: int = 640):
        self.spec = spec
        self.every = max(1, every)
        self.downscale_to = downscale_to
        self.sizes: list[float] = []
        self.n_seen = 0
        self.n_detected = 0
        self.last_ok: bool | None = None   # 最近一次检测是否看到板子

    def offer(self, image) -> float | None:
        """喂一帧（可以随便喂，内部按 every 抽稀）。返回当前的 sqrt(area)。"""
        import cv2

        self.n_seen += 1
        if self.n_seen % self.every:
            return self.sizes[-1] if self.sizes else None
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if w > self.downscale_to:
            sc = self.downscale_to / w
            gray = cv2.resize(gray, (int(w * sc), int(h * sc)),
                              interpolation=cv2.INTER_AREA)
        try:
            from target_detect import detect
            d = detect(gray, self.spec)
        except Exception:  # noqa: BLE001
            return None
        if not d.found or d.image_points is None:
            self.last_ok = False
            return None
        self.last_ok = True
        p = d.image_points.reshape(-1, 2)
        area = float((p[:, 0].ptp()) * (p[:, 1].ptp()))
        self.n_detected += 1
        self.sizes.append(area ** 0.5)
        return self.sizes[-1]

    # -- 指标 --------------------------------------------------
    def depth_span(self) -> float:
        """估计的深度跨度（max/min）。板只在一个距离上时接近 1.0。"""
        if len(self.sizes) < 5:
            return float("nan")
        s = np.asarray(self.sizes)
        return float(np.percentile(s, 95) / max(np.percentile(s, 5), 1e-9))

    def report(self) -> str:
        if len(self.sizes) < 5:
            return (f"  标定板检测 {self.n_detected} 次，样本不足，"
                    f"无法评估深度多样性")
        span = self.depth_span()
        ok = span >= 2.0
        lines = [
            f"  标定板检出 {self.n_detected}/{self.n_seen // self.every} 次",
            f"  估计深度跨度 ≈ {span:.2f}x  "
            f"{'✅ 合格（≥2x）' if ok else '❌ 太小！内参（fx 与畸变）会不可辨识，且残差不报警'}",
        ]
        if not ok:
            lines += [
                "  ⇒ 重采：让标定板走遍**近 / 中 / 远**三档距离。",
                "     受控实验：深度 1.0x 时 fx 偏 +4.17%，3.3x 时只偏 +0.31%。",
            ]
        return "\n".join(lines)


def write_outputs(out: Path, imu_frames, raw: bytes, cam_info: dict,
                  cam_ts_rows: list, board_report: str | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)

    if imu_frames:
        with (out / "imu.txt").open("w") as f:
            f.write(IMU_HEADER + "\n")
            for fr in imu_frames:
                g = fr.six_axis()
                f.write(f"{fr.t_host:.6f} " + " ".join(f"{v:.6f}" for v in g) + "\n")

        with (out / "imu_h7_full.csv").open("w") as f:
            f.write(FULL_HEADER + "\n")
            for fr in imu_frames:
                v = fr.values
                f.write(f"{fr.t_board_ms},{fr.t_host:.6f},"
                        f"{v['gx']:.6f},{v['gy']:.6f},{v['gz']:.6f},"
                        f"{v['ax']:.6f},{v['ay']:.6f},{v['az']:.6f},"
                        f"{v['roll']:.6f},{v['pitch']:.6f},{v['yaw']:.6f},"
                        f"{v['qw']:.6f},{v['qx']:.6f},{v['qy']:.6f},{v['qz']:.6f}\n")

    if raw:
        (out / "imu_raw.bin").write_bytes(raw)

    # image_timestamps.txt —— 标定程序读的就是这个（README 第 5.1 节格式）
    if cam_ts_rows:
        with (out / "image_timestamps.txt").open("w") as f:
            f.write("# filename timestamp_seconds (host monotonic clock, unit: s)\n")
            for r in cam_ts_rows:
                f.write(f"{r['filename']} {r['host_ts']:.9f}\n")
        with (out / "camera_timestamps.csv").open("w") as f:
            f.write("frame,filename,host_monotonic_s,device_ts\n")
            for r in cam_ts_rows:
                f.write(f"{r['frame']},{r['filename']},{r['host_ts']:.9f},"
                        f"{r['device_ts'] if r['device_ts'] is not None else ''}\n")

    meta = {
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_images": len(cam_ts_rows),
        "n_imu_frames": len(imu_frames),
        "imu_raw_bytes": len(raw),
        **cam_info,
    }
    if imu_frames:
        meta["imu_check"] = validate(imu_frames)
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"\n已写入 {out}/")
    print(f"  图像      {meta['n_images']} 张")
    print(f"  IMU       {len(imu_frames)} 帧 ({len(raw)} 字节原始流)")
    if imu_frames:
        st = meta["imu_check"]
        print(f"  采样率    {st['采样率_Hz']:.3f} Hz")
        print(f"  时间戳步长非1的帧数  {st['时间戳步长_非1帧数']}  (应为 0)")
    print(f"  设备时间戳：{'有（camera_timestamps.csv）' if cam_ts_rows and cam_ts_rows[0]['device_ts'] else '无'}")
    if board_report:
        print("\n【采集质量】")
        print(board_report)
    print("\n下一步：")
    print(f"  .venv/bin/python calib/run_calibration.py --session {out} \\")
    print(f"      --pattern <内角点> --square-mm <方格毫米>")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Camera-IMU 数据集采集（工业 U3V 相机 / UVC + H7 IMU）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--serial", default="/dev/ttyACM0", help="IMU 串口")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--camera-source", default="hik",
                    help="hik=工业相机；0/1/...=UVC 索引；none=不接相机")
    ap.add_argument("--width", type=int, default=1624)
    ap.add_argument("--height", type=int, default=1240)
    ap.add_argument("--pixel-format", default="Mono8")
    ap.add_argument("--exposure", type=float, default=None, help="曝光，微秒")
    ap.add_argument("--gain", type=float, default=None, help="增益，dB")
    ap.add_argument("--frame-rate", type=float, default=10.0)
    ap.add_argument("--frames", type=int, default=0,
                    help=">0 进入无窗口自测：自动抓 N 帧后退出")
    ap.add_argument("--frame-interval", type=float, default=0.2)
    ap.add_argument("--auto-interval", type=float, default=0.25)
    ap.add_argument("--pattern", default=None,
                    help="内角点数（如 9x6）。给了就实时评估**深度多样性** —— "
                         "受控实验证明深度跨度 <2x 时内参不可辨识，而残差不报警")
    ap.add_argument("--board-type", default="chessboard",
                    choices=["chessboard", "charuco"])
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--imu-only", action="store_true")
    ap.add_argument("--seconds", type=float, default=0.0)
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
    time.sleep(0.4)
    if imu.n_frames == 0:
        print("❌ 串口打开成功但收不到数据。排查：python3 calib/imu_h7.py --port <串口>")
        collector.stop(); imu.close()
        return 2
    print(f"IMU 已在收数据（{imu.n_frames} 帧 / {imu.n_bytes} 字节）")

    if args.imu_only:
        secs = args.seconds if args.seconds > 0 else 5.0
        print(f"IMU-only 模式，采集 {secs} 秒 ...")
        time.sleep(secs)
        collector.stop(); imu.close()
        write_outputs(out, collector.frames, bytes(imu.raw), {"camera": "none"}, [])
        return 0

    # ── 相机 ───────────────────────────────────────────────
    cam_info = {"camera_source": args.camera_source}
    cap = None
    if args.camera_source.lower() != "none":
        try:
            cap = open_camera(args.camera_source)
            cap.configure(width=args.width, height=args.height,
                          pixel_format=args.pixel_format,
                          exposure_us=args.exposure, gain_db=args.gain,
                          frame_rate=args.frame_rate)
            cam_info.update(cap.info())
            print(f"相机已打开：{cam_info.get('model', cam_info.get('kind'))} "
                  f"{cam_info.get('width')}x{cam_info.get('height')} "
                  f"{cam_info.get('pixel_format', '')}")
        except Exception as e:  # noqa: BLE001
            print(f"❌ {e}")
            print("   可先用 --camera-source none 只采 IMU，或用 --camera-source 0 走 UVC。")
            collector.stop(); imu.close()
            return 2

    cam_ts_rows: list[dict] = []
    frame_iter = cap.frames(None) if cap is not None else None
    idx = 0

    tracker = None
    if args.pattern:
        c, r = (int(v) for v in args.pattern.lower().split("x"))
        tracker = BoardTracker(BoardSpec(args.board_type, (c, r),
                                         args.square_mm / 1000.0))
        print(f"深度多样性监测已开启（板 {args.board_type} {args.pattern}）")

    exp_bad: list = []          # 曝光不合格的帧（见 calib/exposure.py）

    def save_one(frame) -> None:
        nonlocal idx
        import cv2

        name = f"{idx:06d}.png"
        cv2.imwrite(str(out / "images" / name), frame.image)
        cam_ts_rows.append({"frame": idx, "filename": name,
                            "host_ts": frame.host_ts, "device_ts": frame.device_ts})
        # 曝光不合格**当场**报，不要等到标定时才发现整场数据不能用。
        # 2026-09-14 那次采集全废在过曝上，而当时没有任何提示。
        st = exposure_stats(frame.image)
        if st["verdict"] != "ok":
            exp_bad.append((name, st["verdict"]))
            print(f"     ⚠️  {name} 曝光不合格：{st['verdict']} "
                  f"(mean={st['mean']:.1f} 饱和={st['sat_hi']*100:.1f}%)"
                  f"  → 建议把 --exposure {args.exposure:.0f} 改成约 "
                  f"{args.exposure*st['suggest_exposure_factor']:.0f} µs")
        idx += 1
        if tracker is not None:
            tracker.offer(frame.image)

    try:
        if args.frames > 0:
            print(f"无窗口模式：抓 {args.frames} 帧 ...")
            for _ in range(args.frames):
                save_one(next(frame_iter))
                time.sleep(args.frame_interval)
        else:
            import cv2
            print("\n交互模式：s=存一帧  a=自动连拍  q=结束")
            print(f"采集要点：开头静置 3 秒 → 绕三轴快速转动"
                  f"（目标 ≥{TARGET_GYRO_DPS:.0f}°/s）→ 小幅平移 → 板走遍画面")
            auto = False
            last_auto = 0.0
            while True:
                f = next(frame_iter)
                now = time.monotonic()
                if auto and now - last_auto >= args.auto_interval:
                    save_one(f); last_auto = now
                disp = cv2.cvtColor(f.image, cv2.COLOR_GRAY2BGR) \
                    if f.image.ndim == 2 else f.image.copy()
                if tracker is not None:
                    tracker.offer(f.image)
                gdps = collector.recent_gyro_dps()
                warn = bool(idx > 3 and gdps < MIN_GYRO_DPS)
                span = tracker.depth_span() if tracker is not None else float("nan")
                span_txt = (f"  depth={span:.1f}x" if np.isfinite(span) else "")
                if tracker is None:
                    board_txt = ""
                elif tracker.last_ok is None:
                    board_txt = "  board=?"
                else:
                    board_txt = ("  board=OK" if tracker.last_ok
                                 else "  board=NOT FOUND <-- aim at the board")
                txt = (f"saved={idx} auto={'ON' if auto else 'off'}  "
                       f"gyro={gdps:5.1f}deg/s (need>{TARGET_GYRO_DPS:.0f})"
                       f"{span_txt}{board_txt}"
                       + ("  TOO SLOW!" if warn else ""))
                # 曝光同样实时显示：过曝时"板子检不出来"会被误判成对焦/距离问题，
                # 而真正的修法只是调一个曝光参数。
                est = exposure_stats(f.image)
                exp_warn = est["verdict"] != "ok"
                if exp_warn:
                    txt += "  " + ("TOO BRIGHT!" if est["verdict"] == "too_bright"
                                   else "TOO DARK!" if est["verdict"] == "too_dark"
                                   else "LOW CONTRAST!")
                color = (0, 0, 255) if (warn or exp_warn) else (0, 220, 0)
                cv2.putText(disp, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 2)
                cv2.imshow("capture (s save / a auto / q quit)", disp)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("s"):
                    save_one(f)
                elif k == ord("a"):
                    auto = not auto
                    print(f"自动连拍 {'开' if auto else '关'}")
                elif k == ord("q"):
                    break
            cv2.destroyAllWindows()
    finally:
        if cap is not None:
            cap.close()
        collector.stop()
        imu.close()

    rep = tracker.report() if tracker is not None else None
    # 曝光不合格的帧要在**采集结束时**再汇总一次：单帧提示会被滚动输出冲掉，
    # 而它的后果是"整场数据不可用"，必须让人在离开前看到。
    if exp_bad:
        kinds = sorted({v for _, v in exp_bad})
        rep = ((rep + "\n") if rep else "") + "\n".join([
            f"⚠️  曝光不合格 {len(exp_bad)}/{idx} 帧（{'/'.join(kinds)}）",
            "     过曝会让白格与背景一起顶到 255，角点检测直接失效 ——",
            "     这是 2026-09-14 那次采集整场报废的原因。",
            f"     建议把 --exposure 从 {args.exposure:.0f} µs 回调后重采。",
        ])
    write_outputs(out, collector.frames, bytes(imu.raw), cam_info, cam_ts_rows, rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
