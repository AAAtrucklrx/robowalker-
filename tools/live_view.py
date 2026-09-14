#!/usr/bin/env python3
"""实时取景：看相机画面 + 边看边调曝光 + 对焦评分 + 标定板检测。

为什么单独做一个工具
==================================================================
`capture_h7.py` 是**采集**工具：它要连 IMU、要落盘、要管时间戳。但调试阶段
真正高频的需求是"**我现在到底在拍什么**"—— 装完支架朝向变了、镜头被碰了、
曝光不合适，都要靠看画面才能判断。

而且这三个问题里有两个**截图看不出来**：

* **失焦**：离焦的画面肉眼看"挺正常"，只是有点糊。所以我加了**对焦评分**
  （Laplacian 方差）和一个**历史最优值** —— 转对焦环时盯着它涨到最大就对了，
  比肉眼可靠得多。实测 2026-09-14 那次"找不到板子"就是这个原因：
  最大梯度只有 81（正常场景数百），一幅图里 Canny 60/160 的边缘像素是 **0%**。
* **曝光**：均值会骗人（棋盘格黑白各半，均值天然居中），所以用
  `calib/exposure.py` 的饱和占比判据，并给出**可直接照做**的建议。

按键
------------------------------------------------------------------
======  ====================================================
``s``   存一张 PNG 快照（默认 /tmp/live/）
``+``   曝光 ×1.25        ``-``  曝光 ÷1.25
``]``   增益 +2 dB        ``[``  增益 −2 dB
``b``   开关标定板检测
``r``   把"对焦最优值"清零（重新找焦点时用）
``h``   开关帮助文字
``q``   退出（Esc 同）
======  ====================================================

用法::

    .venv/bin/python tools/live_view.py                     # 工业相机，默认参数
    .venv/bin/python tools/live_view.py --exposure 120000
    .venv/bin/python tools/live_view.py --source 0           # 笔记本摄像头
    .venv/bin/python tools/live_view.py --pattern 11x8 --square-mm 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calib"))

from camera import open_camera                    # noqa: E402
from exposure import exposure_stats               # noqa: E402
from target_detect import BoardSpec, detect       # noqa: E402

# 显示缩放：默认**自动探测屏幕分辨率**，能放下就 1:1 显示。
# 为什么不固定缩到 1280：缩放本身就是一次重采样，会把画面**二次模糊**，
# 而对焦判断恰恰依赖画面锐度 —— 为了塞进窗口而牺牲锐度是帮倒忙。
# （本机屏幕 2560x1600，1624x1240 完全放得下，所以默认不缩放。）
DEFAULT_MAX_W = 0          # 0 = 不缩放
DEFAULT_MAX_H = 0


def _screen_size() -> tuple[int, int] | None:
    """尽力探测主屏分辨率；失败返回 None（调用方退化为不缩放）。"""
    import re
    import shutil
    import subprocess

    if shutil.which("xrandr"):
        try:
            out = subprocess.run(["xrandr"], capture_output=True, text=True,
                                 timeout=5).stdout
            for line in out.splitlines():
                if " connected" in line:
                    m = re.search(r"(\d+)x(\d+)\+", line)
                    if m:
                        return int(m.group(1)), int(m.group(2))
        except Exception:                       # noqa: BLE001
            pass
    return None


def focus_score(gray: np.ndarray, crop: bool = True) -> float:
    """对焦评分：Laplacian 方差。越大越清晰。

    只在**中心区域**算（``crop=True``）：中心通常是我们要拍的目标，
    而画面四角如果有强反光/杂物会把全局分数带偏，让对焦环越调越远。
    """
    import cv2

    g = gray
    if crop:
        h, w = g.shape[:2]
        g = g[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    if g.size == 0:
        return 0.0
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _fit(shape, max_w: int, max_h: int) -> float:
    """算显示缩放比。``max_w``/``max_h`` 为 0 表示该方向不限（不缩放）。"""
    h, w = shape[:2]
    cands = [1.0]
    if max_w > 0:
        cands.append(max_w / w)
    if max_h > 0:
        cands.append(max_h / h)
    return min(cands)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="实时取景（曝光 + 对焦 + 标定板检测）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="hik", help="hik / 0 / 1 / ...")
    ap.add_argument("--width", type=int, default=1624)
    ap.add_argument("--height", type=int, default=1240)
    ap.add_argument("--pixel-format", default="Mono8")
    ap.add_argument("--exposure", type=float, default=120000.0, help="曝光 µs")
    ap.add_argument("--gain", type=float, default=10.0, help="增益 dB")
    ap.add_argument("--frame-rate", type=float, default=10.0)
    ap.add_argument("--pattern", default="11x8", help="内角点，如 11x8；none 关闭")
    ap.add_argument("--square-mm", type=float, default=20.0)
    ap.add_argument("--detect-every", type=int, default=2,
                    help="每 N 帧检测一次标定板（1=每帧；SB 单次约 71 ms，"
                         "10 fps 下不建议设 1）")
    ap.add_argument("--save-dir", default="/tmp/live")
    ap.add_argument("--max-w", type=int, default=DEFAULT_MAX_W,
                    help="显示最大宽度（0=不缩放；默认 0）")
    ap.add_argument("--max-h", type=int, default=DEFAULT_MAX_H,
                    help="显示最大高度（0=不缩放；默认 0）")
    args = ap.parse_args()

    import cv2

    # 没显式给 max-w/max-h 就按屏幕来：能放下就 1:1，放不下才缩。
    if args.max_w == 0 and args.max_h == 0:
        scr = _screen_size()
        if scr:
            args.max_w, args.max_h = scr[0] - 80, scr[1] - 120
            print(f"检测到屏幕 {scr[0]}x{scr[1]}，显示上限 "
                  f"{args.max_w}x{args.max_h}（1:1 优先）")

    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)

    spec = None
    if args.pattern.lower() != "none":
        c, r = (int(v) for v in args.pattern.lower().split("x"))
        spec = BoardSpec("chessboard", (c, r), args.square_mm / 1000.0)

    # "hik"/"uvc" 之外的字符串当 UVC 索引处理
    src = args.source
    if src not in ("hik", "uvc") and src.isdigit():
        src = int(src)

    cap = open_camera(src)
    cap.configure(width=args.width, height=args.height,
                  pixel_format=args.pixel_format,
                  exposure_us=args.exposure, gain_db=args.gain,
                  frame_rate=args.frame_rate)
    print("=" * 68)
    print("实时取景已启动")
    print(f"  相机 {src}  曝光 {args.exposure:.0f} µs  增益 {args.gain:.0f} dB")
    if spec:
        print(f"  检测标定板：内角点 {args.pattern}，方格 {args.square_mm} mm")
    print(f"  快照目录 {out}")
    print("=" * 68)
    print("按键：s 存图  +/- 曝光  ]/[ 增益  b 开关板检测  r 重置对焦峰值  h 帮助  q 退出")

    exposure = float(args.exposure)
    gain = float(args.gain)
    detect_on = spec is not None
    show_help = True
    best_focus = 0.0
    n_saved = 0
    n_seen = 0
    _last_det: dict = {"found": False, "pts": None}   # 检测结果缓存（见下）
    fps_t, fps_n, fps = time.monotonic(), 0, 0.0
    scale = None
    focus_dirty = False

    try:
        for f in cap.frames(None):
            img = f.image
            now = time.monotonic()
            fps_n += 1
            if now - fps_t >= 0.5:
                fps = fps_n / (now - fps_t)
                fps_t, fps_n = now, 0

            if scale is None:
                scale = _fit(img.shape, args.max_w, args.max_h)

            fs = focus_score(img)
            if fs > best_focus:
                best_focus = fs
                focus_dirty = True

            st = exposure_stats(img)
            exp_ok = st["verdict"] == "ok"

            # 显示用图（BGR，可缩放）
            disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 \
                else img.copy()

            board_txt, board_ok = "board=off", False
            last_corners = None
            if detect_on and spec is not None:
                # 隔 --detect-every 帧才检测一次：SB 单次 ~71 ms，10 fps 时
                # 每帧都跑就吃掉 71% 的一帧预算（再叠加对焦评分 8 ms +
                # 曝光统计 6 ms 就逼近 100 ms），显示会明显掉帧。
                # 检测结果在间隔内沿用上一帧，视觉上几乎无感。
                if n_seen % max(1, args.detect_every) == 0:
                    d = detect(img, spec, allow_classic=False)
                    _last_det["found"] = bool(d.found and d.image_points is not None)
                    _last_det["pts"] = (d.image_points.copy()
                                        if d.image_points is not None else None)
                n_seen += 1
                if _last_det["found"] and _last_det["pts"] is not None:
                    board_ok = True
                    pts = _last_det["pts"].reshape(-1, 2)
                    if len(pts) == int(np.prod(spec.pattern_size)):
                        cv2.drawChessboardCorners(disp, spec.pattern_size,
                                                  _last_det["pts"], True)
                    span_x = pts[:, 0].max() - pts[:, 0].min()
                    frac = span_x / img.shape[1] * 100
                    board_txt = (f"board=OK {args.pattern} "
                                 f"({len(pts)}pts, {frac:.0f}% width)")
                else:
                    board_txt = "board=NOT FOUND <-- aim at the board"

            if scale < 1.0:
                disp = cv2.resize(disp, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_AREA)

            # ── 叠加文字 ──────────────────────────────────────────────
            lh = 22
            def put(s, row, color=(0, 220, 0)):
                cv2.putText(disp, s, (8, 24 + row * lh),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
                            cv2.LINE_AA)

            put(f"exp={exposure:.0f}us gain={gain:.0f}dB  "
                f"mean={st['mean']:.0f} sat={st['sat_hi']*100:.1f}%  "
                f"fps={fps:.1f}", 0)
            if exp_ok:
                put("exposure OK", 1, (0, 220, 0))
            else:
                sug = exposure * st["suggest_exposure_factor"]
                put(f"EXPOSURE {st['verdict'].upper()} -> try {sug:.0f}us",
                    1, (0, 0, 255))

            # 对焦：显示当前值与"见过的最好值"。
            # 比例 < 0.7 时提示——转对焦环的目标就是让这个比例接近 1。
            ratio = fs / best_focus if best_focus > 0 else 1.0
            fcol = (0, 220, 0) if ratio > 0.9 else \
                   (0, 200, 255) if ratio > 0.7 else (0, 0, 255)
            put(f"focus={fs:6.1f}  best={best_focus:6.1f}  "
                f"{ratio*100:3.0f}% of best", 2, fcol)

            bcol = (0, 220, 0) if board_ok else (0, 0, 255)
            put(board_txt, 3, bcol)

            if show_help:
                put("s save | +/- exposure | ]/[ gain | b detect | "
                    "r reset focus | h help | q quit", 5, (200, 200, 200))

            cv2.imshow("live view (q to quit)", disp)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("s"):
                n_saved += 1
                p = out / f"snap_{n_saved:03d}_exp{exposure:.0f}.png"
                cv2.imwrite(str(p), img)
                print(f"  已存 {p}  （{st['verdict']}, focus={fs:.1f}）")
            elif k in (ord("+"), ord("=")):
                exposure = min(exposure * 1.25, 20_000_000)
                cap.set_exposure(exposure_us=exposure)
                print(f"  曝光 → {exposure:.0f} µs")
            elif k in (ord("-"), ord("_")):
                exposure = max(exposure / 1.25, 1.0)
                cap.set_exposure(exposure_us=exposure)
                print(f"  曝光 → {exposure:.0f} µs")
            elif k == ord("]"):
                gain = min(gain + 2.0, 40.0)
                cap.set_exposure(gain_db=gain)
                print(f"  增益 → {gain:.0f} dB")
            elif k == ord("["):
                gain = max(gain - 2.0, 0.0)
                cap.set_exposure(gain_db=gain)
                print(f"  增益 → {gain:.0f} dB")
            elif k == ord("b"):
                detect_on = not detect_on
                print(f"  标定板检测 {'开' if detect_on else '关'}")
            elif k == ord("r"):
                best_focus = 0.0
                print("  对焦峰值已重置")
            elif k == ord("h"):
                show_help = not show_help
    finally:
        cv2.destroyAllWindows()
        cap.close()

    print(f"\n结束。共存 {n_saved} 张到 {out}")
    print(f"  见过的最好对焦分 = {best_focus:.1f}")
    print("  最后曝光/增益：" f"{exposure:.0f} µs / {gain:.0f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
