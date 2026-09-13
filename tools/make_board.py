#!/usr/bin/env python3
"""生成**物理尺寸精确**的可打印标定板（棋盘格 / ChArUco）。

为什么需要这个脚本
------------------------------------------------------------------
任务书 5.3 与 README 都强调：**``square_size`` 与实物不符，内参仍然正确，
但平移外参的尺度会整体错。** 而"打印出来尺寸不对"是这条路上最常见的坑，
原因往往是：

* 打印机默认"适合页面 / 缩放以填充"→ 整个板子被缩放了几个百分点；
* 用截图或 PNG 打印 → 分辨率换算引入误差。

本脚本直接输出 **PDF**（矢量，尺寸以点为单位精确指定），
你只要在打印对话框里选 **"实际大小 / 100% / 无缩放"**，
再拿尺子核一下页面尺寸，``square_size`` 就是可信的。

用法
------------------------------------------------------------------
    # 默认：10x7 方格（= 9x6 内角点），方格 25 mm -> 250x175 mm，A4 横向放得下
    .venv/bin/python tools/make_board.py --out boards/chess_9x6_25mm.pdf

    # 带 100 mm 校验尺（强烈建议，用来确认打印机没缩放）
    .venv/bin/python tools/make_board.py --with-ruler --out boards/chess_ruler.pdf

    # ChArUco（部分遮挡/出画也能检测，比纯棋盘格鲁棒，推荐进阶使用）
    .venv/bin/python tools/make_board.py --type charuco --out boards/charuco.pdf

打印后务必做这一步
------------------------------------------------------------------
    用尺子量一个方格的边长。**必须是 25.0 mm**（或你指定的值）。
    如果量出来是 24.2 或 25.8，请回打印机设置重打，
    **不要**把实测值填进配置——那是在给打印机误差做补偿，会污染标定。

得到内角点数
------------------------------------------------------------------
``pattern_size`` 填的是**内角点**数，不是方格数：
方格 10x7 → 内角点 **9x6**。棋盘格与 ChArUco 都一样。
"""
from __future__ import annotations

import argparse
from pathlib import Path

MM_PER_INCH = 25.4


def _fig(w_mm: float, h_mm: float, dpi: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(w_mm / MM_PER_INCH, h_mm / MM_PER_INCH), dpi=dpi)
    return fig, plt


def _draw_ruler(ax, x0, y0, length_mm, tick_mm=10, label=True):
    """在 (x0,y0) 处画一条 length_mm 长的校验尺（坐标单位 mm）。"""
    ax.plot([x0, x0 + length_mm], [y0, y0], color="black", linewidth=0.8)
    n = int(length_mm // tick_mm)
    for i in range(n + 1):
        x = x0 + i * tick_mm
        h = 3.0 if i % 5 == 0 else 1.5
        ax.plot([x, x], [y0, y0 + h], color="black", linewidth=0.6)
        if label and i % 5 == 0:
            ax.text(x, y0 + h + 0.8, str(i * tick_mm), fontsize=4,
                    ha="center", va="bottom")
    if label:
        # 板面上的文字一律用 ASCII：matplotlib 默认字体没有 CJK 字形，
        # 写中文会渲染成方框（tofu）。中文说明放在终端输出与 README 里。
        ax.text(x0 + length_mm + 1.5, y0, f"{length_mm} mm  <- verify",
                fontsize=5, ha="left", va="center")


def make_chessboard(cols: int, rows: int, square_mm: float, out_pdf: Path,
                    out_png: Path | None = None, dpi: int = 300,
                    with_ruler: bool = False, margin_mm: float = 14.0):
    """cols x rows 个**方格**（内角点 = (cols-1) x (rows-1)）。"""
    board_w, board_h = cols * square_mm, rows * square_mm
    page_w = board_w + (margin_mm if with_ruler else 0)
    page_h = board_h + (margin_mm if with_ruler else 0)

    fig, plt = _fig(page_w, page_h, dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w)
    ax.set_ylim(0, page_h)
    ax.set_axis_off()

    # 画方格。让**左上角**是黑格（约定俗成，OpenCV 不关心，但方便你描述）
    from matplotlib.patches import Rectangle

    for r in range(rows):
        for c in range(cols):
            yy = (rows - 1 - r) * square_mm     # 图像行 -> 页面 y（自下而上）
            if (r + c) % 2 == 0:
                ax.add_patch(Rectangle((c * square_mm, yy), square_mm, square_mm,
                                       facecolor="black", edgecolor="none"))

    # 板子外围加一条细边框，方便对齐与裁切（细到不影响检测）
    ax.add_patch(Rectangle((0, 0), board_w, board_h, facecolor="none",
                           edgecolor="black", linewidth=0.4))

    if with_ruler:
        _draw_ruler(ax, 0, board_h + 4, min(100.0, page_w - 6))
        ax.text(0, board_h + 9,
                f"CHESSBOARD {cols}x{rows} squares (inner corners {cols-1}x{rows-1})"
                f"  |  square = {square_mm} mm  |  board {board_w:.0f}x{board_h:.0f} mm",
                fontsize=5, ha="left", va="bottom")
        ax.text(0, board_h + 11.5,
                "PRINT AT 100% / ACTUAL SIZE (no 'fit to page'). "
                "Then measure one square edge to confirm.",
                fontsize=5, ha="left", va="bottom")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf")
    if out_png:
        fig.savefig(out_png, dpi=dpi)
    plt.close(fig)

    print(f"✅ {out_pdf}")
    print(f"   页面尺寸 {page_w:.1f} x {page_h:.1f} mm   "
          f"({'A4 放得下' if page_w <= 297 and page_h <= 210 else '注意：可能超 A4'})")
    print(f"   方格 {cols} x {rows}（{square_mm} mm） -> 内角点 {cols-1} x {rows-1}")
    print(f"   配置里填：pattern_size: [{cols-1}, {rows-1}]   "
          f"square_size: {square_mm/1000.0}")
    if with_ruler:
        print(f"   右侧有 100 mm 校验尺，打印后量一下确认没有被缩放")
    if out_png:
        print(f"✅ {out_png}")


def make_charuco(squares_x: int, squares_y: int, square_mm: float,
                 marker_ratio: float, out_pdf: Path, out_png: Path | None = None,
                 dpi: int = 600, dict_name: str = "DICT_5X5_100",
                 with_ruler: bool = False, margin_mm: float = 14.0):
    """ChArUco 板：方格 + ArUco。部分出画/遮挡也能检测，比纯棋盘格鲁棒。"""
    import cv2
    import numpy as np

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    marker_mm = square_mm * marker_ratio
    # 兼容 OpenCV 4.6 (CharucoBoard_create) 与 4.7+ (CharucoBoard)
    if hasattr(aruco, "CharucoBoard"):
        board = aruco.CharucoBoard((squares_x, squares_y), square_mm, marker_mm,
                                   dictionary)
    else:
        board = aruco.CharucoBoard_create(squares_x, squares_y, square_mm,
                                          marker_mm, dictionary)
    board_w, board_h = squares_x * square_mm, squares_y * square_mm
    px_per_mm = dpi / MM_PER_INCH
    size = (int(round(board_w * px_per_mm)), int(round(board_h * px_per_mm)))
    img = None
    for fn in ("generateImage", "draw"):
        if hasattr(board, fn):
            try:
                img = getattr(board, fn)(size, marginSize=0)
            except TypeError:
                img = getattr(board, fn)(size)
            break
    if img is None:
        raise RuntimeError("这个 OpenCV 版本的 CharucoBoard 不支持生成图像")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    page_w = board_w + (margin_mm if with_ruler else 0)
    page_h = board_h + (margin_mm if with_ruler else 0)
    fig, plt = _fig(page_w, page_h, dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w)
    ax.set_ylim(0, page_h)
    ax.set_axis_off()
    ax.imshow(img, cmap="gray", extent=[0, board_w, 0, board_h],
              interpolation="nearest", vmin=0, vmax=255)
    if with_ruler:
        _draw_ruler(ax, 0, board_h + 4, min(100.0, page_w - 6))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf")
    if out_png:
        fig.savefig(out_png, dpi=dpi)
    plt.close(fig)

    print(f"✅ {out_pdf}")
    print(f"   页面尺寸 {page_w:.1f} x {page_h:.1f} mm")
    print(f"   ChArUco {squares_x} x {squares_y} 方格（{square_mm} mm），"
          f"marker = {marker_mm:.2f} mm，字典 {dict_name}")
    print(f"   配置里填：pattern_size: [{squares_x-1}, {squares_y-1}]   "
          f"square_size: {square_mm/1000.0}")
    if out_png:
        print(f"✅ {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="生成物理尺寸精确的可打印标定板",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--type", choices=["chessboard", "charuco"],
                    default="chessboard")
    ap.add_argument("--cols", type=int, default=10, help="横向**方格**数")
    ap.add_argument("--rows", type=int, default=7, help="纵向**方格**数")
    ap.add_argument("--square-mm", type=float, default=25.0, help="方格边长 (mm)")
    ap.add_argument("--marker-ratio", type=float, default=0.75,
                    help="ChArUco: marker / 方格 边长比")
    ap.add_argument("--dict", default="DICT_5X5_100", help="ChArUco 字典")
    ap.add_argument("--out", required=True, help="输出 PDF 路径")
    ap.add_argument("--png", action="store_true", help="同时输出 PNG")
    ap.add_argument("--with-ruler", action="store_true",
                    help="附带 100 mm 校验尺（强烈建议）")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    png = out.with_suffix(".png") if args.png else None

    if args.type == "chessboard":
        make_chessboard(args.cols, args.rows, args.square_mm, out, png,
                        dpi=args.dpi, with_ruler=args.with_ruler)
    else:
        if args.cols % 2 == 0 or args.rows % 2 == 0:
            print("⚠️  ChArUco 建议方格数为奇数 x 偶数（如 11x8），检测更稳")
        make_charuco(args.cols, args.rows, args.square_mm, args.marker_ratio,
                     out, png, dpi=max(args.dpi, 600), dict_name=args.dict,
                     with_ruler=args.with_ruler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
