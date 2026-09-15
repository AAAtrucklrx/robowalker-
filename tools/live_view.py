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

# 大于目标规格的常见子规格，用于"还需退后多少"的提示（由大到小试）
# 探测"能检出的最大子规格"时按这个顺序试（覆盖得开，不是简单降序）。
# 顺序很关键：板子被裁切时真正能检出的是偏小的规格；若从 10x7 一路顺序
# 往下试，还没轮到 6x4 就用完 max_tries 了（实测踩过），得到的提示会变成
# "连 4x3 都检不出"，反而误导。
_SUB_PATTERNS = [(11, 8), (9, 6), (8, 5), (6, 4), (4, 3), (5, 4), (10, 7)]

# 显示缩放：默认**自动探测屏幕分辨率**，能放下就 1:1 显示。
# 为什么不固定缩到 1280：缩放本身就是一次重采样，会把画面**二次模糊**，
# 而对焦判断恰恰依赖画面锐度 —— 为了塞进窗口而牺牲锐度是帮倒忙。
# （本机屏幕 2560x1600，1624x1240 完全放得下，所以默认不缩放。）
DEFAULT_MAX_W = 0          # 0 = 不缩放
DEFAULT_MAX_H = 0

# Ctrl-C / SIGTERM 时置位，让主循环干净退出并**释放相机**。
# 不装这个的话，被 Ctrl-C 打断的进程会把相机留在 USB 假死状态
# （见 calib/camera.py 里 usb_reset_camera 的说明）。
_STOP = {"flag": False}


def _install_signal_handlers() -> None:
    import signal

    def _on_sig(signum, _frame):
        _STOP["flag"] = True
        print(f"\n收到信号 {signum}，正在退出并释放相机 ...", flush=True)

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _on_sig)
        except (ValueError, OSError):
            pass


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


def _norm_cv(k: int) -> str | None:
    """把 ``cv2.waitKey`` 的返回值规范化成按键名。"""
    if k in (-1, 0xFF):
        return None
    if k == 27:
        return "Escape"
    try:
        return chr(k)
    except ValueError:
        return None


def _norm_tk(keysym: str) -> str | None:
    """把 tkinter 的 keysym 规范化成与 :func:`_norm_cv` 同一套名字。"""
    m = {"plus": "+", "equal": "=", "minus": "-", "underscore": "_",
         "bracketright": "]", "bracketleft": "[", "Escape": "Escape",
         "KP_Add": "+", "KP_Subtract": "-"}
    if keysym in m:
        return m[keysym]
    return keysym if len(keysym) == 1 else None


def _probe_largest_pattern(img, spec, want, max_tries: int = 4):
    """目标规格没检出时，找**能检出的最大子规格**，用于提示"还要退后多少"。

    为什么不自己估方格像素：试过"扫描线游程中位数"，在有透视/倾斜时
    严重失真（实测 131 px vs 真值 204 px）。用**真正的检测器**去试几个
    小规格，结果一定可信 —— 代价是每次约 70 ms，所以调用方要抽稀
    （见 ``--probe-every``）。

    返回 ``(cols, rows, image_points)`` 或 ``None``。
    """
    from target_detect import BoardSpec, detect as _detect

    tried = 0
    for (c, r) in _SUB_PATTERNS:
        if (c, r) == tuple(want):
            continue
        if tried >= max_tries:
            break
        tried += 1
        try:
            d = _detect(img, BoardSpec("chessboard", (c, r), spec.square_size),
                        allow_classic=False)
        except Exception:                       # noqa: BLE001
            continue
        if d.found and d.image_points is not None:
            return (c, r, d.image_points)
    return None


def _backoff_hint(img_w: int, want_cols: int, sub) -> str:
    """由检出的子规格算"还需把距离拉大多少倍"。"""
    c, r, pts = sub
    p = pts.reshape(-1, 2)
    if c < 2:
        return ""
    cell_px = (p[:, 0].max() - p[:, 0].min()) / (c - 1)
    # want_cols 是**内角点**数 → 板子横向有 want_cols+1 个方格
    # （11 个内角点 = 12 个方格）。要求整板加边距落在画面 90% 宽度内。
    need_px = img_w * 0.90 / max(1, want_cols + 1)
    # ⚠️ 必须返回**英文**：cv2.putText 用的是 Hershey 字体，只能画 ASCII，
    # 中文会原样变成一串 "?"（实测窗口里显示成
    # "board=NOT FOUND ?????? 4x3 ??????..."，完全不可读）。
    # 详细的中文解释走终端（见 main 里的打印）。
    if cell_px <= need_px:
        return f"(found {c}x{r}, adjust framing)"
    return f"(max {c}x{r}, back off ~{cell_px / need_px:.1f}x)"


class CvViewer:
    """OpenCV HighGUI 显示。

    ⚠️ 本机 OpenCV 4.11 的 GUI 后端是 **QT5**（``GTK+: NO``，换不了），
    实测在"两帧之间间隔较长"时会出现 **futex 死锁**：窗口变黑、关不掉、
    进程 28 线程 81% CPU 卡在 ``futex_do_wait``。所以默认不用它。
    """

    name = "cv"

    def __init__(self, title: str, gui_normal: bool = False):
        import cv2
        self.cv2 = cv2
        self.title = title
        self.win = title
        if gui_normal:
            cv2.namedWindow(self.win, cv2.WINDOW_GUI_NORMAL)

    def show(self, bgr) -> None:
        self.cv2.imshow(self.win, bgr)

    def key(self) -> str | None:
        return _norm_cv(self.cv2.waitKey(1) & 0xFF)

    def close(self) -> None:
        try:
            self.cv2.destroyWindow(self.win)
        except Exception:                       # noqa: BLE001
            pass
        try:
            self.cv2.destroyAllWindows()
        except Exception:                       # noqa: BLE001
            pass


class TkViewer:
    """Tkinter + Pillow 显示（**默认**）。

    为什么默认换成 tkinter
    ------------------------------------------------------------------
    OpenCV 的 Qt 后端会死锁（见 :class:`CvViewer`），而 tkinter 是纯 Python
    事件循环，``update()`` 由我们自己控制，不依赖 OpenCV 的 GUI 子系统，
    也就没有那类"Qt 事件循环被饿死 → 窗口变黑 → 关不掉"的问题。

    两个必须注意的点：

    1. ``ImageTk.PhotoImage`` **必须保住引用**（``self._photo``）。不保引用的话
       对象被 GC 掉，窗口会显示成**空白/黑色** —— 这正好也是"黑屏"的一种成因。
    2. 关闭窗口要绑 ``WM_DELETE_WINDOW``，否则点 × 只会销毁窗口而循环还在跑。
    """

    name = "tk"

    def __init__(self, title: str, scale: float = 1.0):
        import tkinter as tk
        from PIL import Image, ImageTk
        self.tk, self.Image, self.ImageTk = tk, Image, ImageTk
        self.scale = scale
        self.root = tk.Tk()
        self.root.title(title)
        self.label = tk.Label(self.root, borderwidth=0)
        self.label.pack()
        self._photo = None            # ← 见类文档第 1 点，必须持有
        self._photo_size = None
        self._key: str | None = None
        self.closed = False
        self.root.bind("<Key>", self._on_key)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.focus_force()

    def _on_key(self, event) -> None:
        self._key = _norm_tk(event.keysym)

    def _on_close(self) -> None:
        self.closed = True

    def show(self, bgr) -> None:
        from PIL import Image
        if self.scale < 1.0:
            import cv2
            bgr = cv2.resize(bgr, None, fx=self.scale, fy=self.scale,
                             interpolation=cv2.INTER_AREA)
        # np.ascontiguousarray 是必要的：bgr[:, :, ::-1] 是**负步长视图**，
        # PIL 的 fromarray 直接吃它可能拿到错位的数据。
        arr = np.ascontiguousarray(bgr[:, :, ::-1])       # BGR→RGB
        im = Image.fromarray(arr)

        if self._photo is None or im.size != self._photo_size:
            # 只在第一帧（或尺寸变了）创建 Tk 图像。
            #
            # ⚠️ 这里是**内存泄漏的关键**：早先的实现每帧都
            # ``ImageTk.PhotoImage(im)`` 新建一个 Tk 图像。Tk 的
            # ``image delete`` 是**延迟**到解释器空闲时才执行的，而我们的
            # 循环几乎不让它空闲 —— 于是 Tk 图像不断堆积。实测 RSS
            # 30 s 内从 385 MB 涨到 471 MB（约 2.9 MB/s），几分钟就能吃掉
            # 几个 GB，最后显示层自己出错退出。
            # 正确做法是**只建一次**，之后用 ``paste`` 原地更新内容。
            self._photo = self.ImageTk.PhotoImage(im)
            self._photo_size = im.size
            self.label.configure(image=self._photo)
        else:
            self._photo.paste(im)          # 原地更新，不新建 Tk 图像

        try:
            self.root.update_idletasks()
            self.root.update()
        except self.tk.TclError as e:
            # 别静默退出 —— 之前这里默默置 closed，导致"跑一会儿自己就没了"
            # 却完全不知道原因。
            print(f"  ⚠️  显示刷新失败（TclError: {e}），窗口可能已关闭",
                  flush=True)
            self.closed = True

    def key(self) -> str | None:
        k, self._key = self._key, None
        return k

    def close(self) -> None:
        try:
            self.root.destroy()
        except Exception:                       # noqa: BLE001
            pass


class NullViewer:
    """无窗口（``--display none``）：只跑逻辑，供自动化测试用。"""

    name = "none"

    def __init__(self, title: str, **_):
        self.closed = False

    def show(self, bgr) -> None:
        pass

    def key(self) -> str | None:
        return None

    def close(self) -> None:
        pass


def _make_viewer(kind: str, title: str, scale: float):
    if kind == "tk":
        return TkViewer(title, scale=scale)
    if kind == "cv":
        return CvViewer(title)
    return NullViewer(title)


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
    ap.add_argument("--detect-scale", type=float, default=0.5,
                    help="检测前把图缩小到几倍（0.5=半分辨率）。SB 在 "
                         "1624x1240 上要 70~250 ms，缩小 2 倍约快 4 倍；"
                         "角点坐标会换算回原图，不影响绘制精度")
    ap.add_argument("--probe-every", type=float, default=3.0,
                    help="目标规格没检出时，每隔几秒探测一次"
                         "'能检出的最大子规格'用于提示退后倍数（0=关闭）")
    ap.add_argument("--detect-every", type=int, default=2,
                    help="每 N 帧检测一次标定板（1=每帧；SB 单次约 71 ms，"
                         "10 fps 下不建议设 1）")
    ap.add_argument("--save-dir", default="/tmp/live")
    ap.add_argument("--max-w", type=int, default=DEFAULT_MAX_W,
                    help="显示最大宽度（0=不缩放；默认 0）")
    ap.add_argument("--max-h", type=int, default=DEFAULT_MAX_H,
                    help="显示最大高度（0=不缩放；默认 0）")
    ap.add_argument("--log-every", type=float, default=5.0,
                    help="每几秒往终端打一行心跳（fps/帧号/内存）。"
                         "排查'程序跑到一半死机'时靠它留下最后状态；0=关闭")
    ap.add_argument("--display", default="tk", choices=["tk", "cv", "none"],
                    help="显示后端。tk(默认)=Tkinter，绕开 OpenCV 的 Qt 后端；"
                         "cv=OpenCV HighGUI（本机实测会 futex 死锁，慎用）；"
                         "none=无窗口")
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

    _install_signal_handlers()

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
    print(f"  显示后端 {args.display}")
    if spec:
        print(f"  检测标定板：内角点 {args.pattern}，方格 {args.square_mm} mm")
    print(f"  快照目录 {out}")
    print("=" * 68)
    print("按键：s 存图  +/- 曝光  ]/[ 增益  b 开关板检测  r 重置对焦峰值  h 帮助  q 退出")
    print("（窗口内只用英文：OpenCV 的 putText 画不了中文，中文提示会走这个终端）")

    # 显示缩放比：这里就按**配置的分辨率**算出来，因为 TkViewer 构造时就要用。
    scale = _fit((args.height, args.width), args.max_w, args.max_h)
    viewer = _make_viewer(args.display, "live view (q / Esc to quit)", scale)

    exposure = float(args.exposure)
    gain = float(args.gain)
    detect_on = spec is not None
    show_help = True
    best_focus = 0.0
    n_saved = 0
    n_seen = 0
    _last_det: dict = {"found": False, "pts": None}   # 检测结果缓存（见下）
    _last_probe: dict = {"t": 0.0, "hint": "", "sub": None}   # 子规格探测 + 提示
    fps_t, fps_n, fps = time.monotonic(), 0, 0.0
    focus_dirty = False
    n_frames = 0
    log_t = time.monotonic()

    try:
        for f in cap.frames(None, latest=True):
            img = f.image
            now = time.monotonic()
            fps_n += 1
            if now - fps_t >= 0.5:
                fps = fps_n / (now - fps_t)
                fps_t, fps_n = now, 0

            fs = focus_score(img)
            if fs > best_focus:
                best_focus = fs
                focus_dirty = True

            st = exposure_stats(img)
            exp_ok = st["verdict"] == "ok"

            # 检测用的小图（只做一次缩放，下面检测和探测共用）
            _small = (cv2.resize(img, None, fx=args.detect_scale,
                                 fy=args.detect_scale,
                                 interpolation=cv2.INTER_AREA)
                      if args.detect_scale < 1.0 else img)

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
                    d = detect(_small, spec, allow_classic=False)
                    _last_det["found"] = bool(d.found and d.image_points is not None)
                    # 角点坐标要**换回原图尺度**，否则画上去会缩在左上角
                    _last_det["pts"] = ((d.image_points / args.detect_scale).copy()
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
                    # 目标规格没检出时，抽稀地探测"能检出的最大子规格"，
                    # 直接告诉用户"还要退后多少" —— 否则只能靠反复试。
                    board_txt = "board=NOT FOUND <-- aim at the board"
                    if (args.probe_every > 0
                            and now - _last_probe["t"] >= args.probe_every):
                        _last_probe["t"] = now
                        sub = _probe_largest_pattern(_small, spec,
                                                     spec.pattern_size)
                        if sub is not None:
                            # 坐标换回原图尺度，下面要用它画角点
                            sub = (sub[0], sub[1], sub[2] / args.detect_scale)
                        _last_probe["sub"] = sub
                        if sub:
                            _last_probe["hint"] = _backoff_hint(
                                img.shape[1], spec.pattern_size[0], sub)
                        else:
                            _last_probe["hint"] = "(nothing found: aim/focus/exposure)"
                        # 中文详解打到终端（窗口里画不了中文）
                        if _last_probe["hint"] != _last_probe.get("last_printed"):
                            _last_probe["last_printed"] = _last_probe["hint"]
                            if sub:
                                print(f"  [提示] 板子未完整检出：最大只认出 "
                                      f"{sub[0]}x{sub[1]}，请把相机/板子距离"
                                      f"拉大或调整构图（见窗口 board= 那行）",
                                      flush=True)
                            else:
                                print("  [提示] 连 4x3 都认不出：请先对准板子、"
                                      "拧对焦环（看 focus= 冲到最大）、"
                                      "并让 exposure 变绿", flush=True)
                    # 把"能检出的最大子区域"的角点也画出来 —— 这是最直观的
                    # 瞄准反馈：用户能直接看到相机锁住了板子的哪一块、还差多少。
                    _sub = _last_probe.get("sub")
                    if _sub is not None:
                        _c, _r, _pts = _sub
                        _pts = np.asarray(_pts, dtype=np.float32).reshape(-1, 1, 2)
                        if len(_pts) == _c * _r:
                            cv2.drawChessboardCorners(disp, (_c, _r), _pts, True)
                    if _last_probe["hint"]:
                        board_txt += "  " + _last_probe["hint"]

            if scale < 1.0:
                disp = cv2.resize(disp, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_AREA)

            # ── 叠加文字 ──────────────────────────────────────────────
            # 文字放大到 0.75 —— 用户是隔着一段距离手持看屏幕的，
            # 0.55 太小看不清（实测反馈"看不出在说什么"）。
            lh = 30
            def put(s, row, color=(0, 220, 0)):
                cv2.putText(disp, s, (10, 32 + row * lh),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2,
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

            n_frames += 1
            # 心跳：万一后面挂死，这行就是"最后活着的状态"（配合 --log-every）
            if args.log_every > 0 and now - log_t >= args.log_every:
                log_t = now
                try:
                    rss = 0
                    with open("/proc/self/status") as fh:
                        for ln in fh:
                            if ln.startswith("VmRSS"):
                                rss = int(ln.split()[1])
                                break
                except OSError:
                    rss = 0
                print(f"  [心跳] 帧={n_frames} fps={fps:4.1f} "
                      f"板={'OK' if board_ok else '--'} focus={fs:6.1f} "
                      f"exp={exposure:.0f} RSS={rss/1024:.0f}MB", flush=True)

            viewer.show(disp)
            k = viewer.key()
            if k in ("q", "Escape") or getattr(viewer, "closed", False) \
                    or _STOP["flag"]:
                break
            elif k == "s":
                n_saved += 1
                p = out / f"snap_{n_saved:03d}_exp{exposure:.0f}.png"
                cv2.imwrite(str(p), img)
                print(f"  已存 {p}  （{st['verdict']}, focus={fs:.1f}）")
            elif k in ("+", "="):
                exposure = min(exposure * 1.25, 20_000_000)
                cap.set_exposure(exposure_us=exposure)
                print(f"  曝光 → {exposure:.0f} µs")
            elif k in ("-", "_"):
                exposure = max(exposure / 1.25, 1.0)
                cap.set_exposure(exposure_us=exposure)
                print(f"  曝光 → {exposure:.0f} µs")
            elif k == "]":
                gain = min(gain + 2.0, 40.0)
                cap.set_exposure(gain_db=gain)
                print(f"  增益 → {gain:.0f} dB")
            elif k == "[":
                gain = max(gain - 2.0, 0.0)
                cap.set_exposure(gain_db=gain)
                print(f"  增益 → {gain:.0f} dB")
            elif k == "b":
                detect_on = not detect_on
                print(f"  标定板检测 {'开' if detect_on else '关'}")
            elif k == "r":
                best_focus = 0.0
                print("  对焦峰值已重置")
            elif k == "h":
                show_help = not show_help
    finally:
        # 顺序很重要：先关窗口再放相机。反过来若关窗口卡住，相机就漏了。
        viewer.close()
        cap.close()

    print(f"\n结束。共存 {n_saved} 张到 {out}")
    print(f"  见过的最好对焦分 = {best_focus:.1f}")
    print("  最后曝光/增益：" f"{exposure:.0f} µs / {gain:.0f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
