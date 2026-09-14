#!/usr/bin/env python3
"""GUI 稳定性压力测试：找出让 OpenCV Qt 窗口不死锁的配置。

背景：用户实测 live_view 跑 1~2 分钟后窗口变黑、关不掉，进程 wchan=futex_do_wait、
28 线程、81% CPU —— 典型的 OpenCV **Qt5 后端 + pthreads 并行池**死锁。
本机 OpenCV 4.11 构建是 `GUI: QT5`、`GTK+: NO`，所以换后端这条路走不通，
只能调参数。

本脚本把 live_view 的主循环抽出来跑固定时长，轮流试不同配置，
报告每档的实际 fps 与是否存活。

用法::
    .venv/bin/python tools/stress_gui.py --seconds 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calib"))


def run(cv2, np, cap, spec, seconds, threads, gui_normal, detect_every, label):
    if threads > 0:
        cv2.setNumThreads(threads)
    win = f"stress {label}"
    if gui_normal:
        cv2.namedWindow(win, cv2.WINDOW_GUI_NORMAL)
    from exposure import exposure_stats
    from target_detect import detect
    from live_view import focus_score

    it = cap.frames(None, timeout_us=3_000_000)
    t0 = time.monotonic()
    n = 0
    last = t0
    fps_marks = []
    while time.monotonic() - t0 < seconds:
        f = next(it)
        img = f.image
        exposure_stats(img)
        focus_score(img)
        if n % max(1, detect_every) == 0:
            detect(img, spec, allow_classic=False)
        disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(disp, f"{label} n={n}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        cv2.imshow(win, disp)
        cv2.waitKey(1)
        n += 1
        now = time.monotonic()
        if now - last >= 5.0:
            fps_marks.append(n / (now - t0))
            print(f"    [{label}] t={now-t0:5.1f}s  n={n}  累计fps={n/(now-t0):5.2f}",
                  flush=True)
            last = now
    cv2.destroyWindow(win)
    dt = time.monotonic() - t0
    print(f"  ✅ [{label}] 存活完成 {n} 帧 / {dt:.1f}s = {n/dt:.2f} fps", flush=True)
    return n / dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--exposure", type=float, default=50000)
    ap.add_argument("--gain", type=float, default=15)
    ap.add_argument("--only", default=None, help="只跑指定档（标签子串）")
    args = ap.parse_args()

    import cv2
    import numpy as np
    from camera import open_camera
    from target_detect import BoardSpec

    spec = BoardSpec("chessboard", (11, 8), 0.020)

    # 每档：(标签, 线程数, 用 WINDOW_GUI_NORMAL, 检测间隔)
    configs = [
        ("baseline-24t", 24, False, 2),
        ("t1-guinormal", 1, True, 3),
        ("t4-guinormal", 4, True, 3),
        ("t1-default", 1, False, 3),
    ]
    if args.only:
        configs = [c for c in configs if args.only in c[0]]

    print("=" * 68)
    print(f"GUI 稳定性压力测试（每档 {args.seconds:.0f} s）")
    print("=" * 68)
    for label, threads, gui_normal, every in configs:
        print(f"\n▶ {label}: threads={threads} gui_normal={gui_normal} "
              f"detect_every={every}", flush=True)
        cap = None
        try:
            cap = open_camera("hik")
            cap.configure(width=1624, height=1240, pixel_format="Mono8",
                          exposure_us=args.exposure, gain_db=args.gain,
                          frame_rate=10)
            run(cv2, np, cap, spec, args.seconds, threads, gui_normal, every, label)
        except BaseException as e:                      # noqa: BLE001
            print(f"  ❌ [{label}] {type(e).__name__}: {e}", flush=True)
        finally:
            try:
                if cap is not None:
                    cap.close()
            except Exception:                           # noqa: BLE001
                pass
            time.sleep(1.0)
    print("\n全部档位跑完", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
