#!/usr/bin/env python3
"""取景显示后端：Tkinter（默认，稳）与 OpenCV HighGUI（会死锁，慎用）。

为什么单独抽成模块
==================================================================
本机 OpenCV 4.11 的 GUI 后端是 **QT5**（``GTK+: NO``，换不了），配 pthreads
并行框架时会**死锁**：窗口变黑、关不掉，进程 28 线程 81% CPU 卡在
``futex_do_wait``（实测）。Tkinter 是纯 Python 事件循环，没有这个问题。

实时取景（``live_view``）和采集（``capture_h7``）都需要窗口，所以把显示抽到
这里共用，免得两份实现各踩一遍坑。两个必须注意的点在这里统一处理：

1. ``ImageTk.PhotoImage`` **必须保住引用**，否则被 GC 后窗口显示成**黑屏**；
2. 每帧新建 PhotoImage 会**内存泄漏**（Tk 的 image delete 延迟执行）——
   实测 2.9 MB/s，几分钟吃掉几个 GB。必须只建一次、之后 ``paste`` 原地更新。

用法::

    from viewer import make_viewer
    v = make_viewer("tk", "标题", scale=1.0)
    v.show(bgr)
    k = v.key()        # 'q' / '+' / '-' / ']' / '[' / 'Escape' / None
    v.close()
"""
from __future__ import annotations

import re
import shutil
import subprocess

import numpy as np


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

def make_viewer(kind: str, title: str, scale: float):
    if kind == "tk":
        return TkViewer(title, scale=scale)
    if kind == "cv":
        return CvViewer(title)
    return NullViewer(title)
