#!/usr/bin/env python3
"""找板子：不用弹窗，连拍一组图后自动分析**哪一帧里有标定板、是什么规格**。

什么时候用
------------------------------------------------------------------
调试时遇到"相机明明对着板子，程序就是 NOT FOUND"，但又不方便盯着 GUI 窗口
（或者窗口根本弹不出来）。这个脚本让你**一边慢慢转相机一边按回车拍一组**，
它再把每一帧都过一遍常见规格，直接告诉你结果。

它同时会存图，所以就算全都没检出，你也能打开图片确认相机到底在拍什么。

用法::

    .venv/bin/python tools/hunt_board.py --out /tmp/hunt
    # 出现提示后：慢慢转动相机（每个方向停 1 秒），按回车拍一张；拍够 N 张自动结束
    # 也可以 --auto 让它自己按固定间隔拍，你只管匀速转
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calib"))

from camera import open_camera  # noqa: E402
from exposure import exposure_stats, format_hint  # noqa: E402
from target_detect import BoardSpec, detect  # noqa: E402

# 常见规格（内角点）。含用户说的 12x9 方格 -> 11x8 内角点，以及"12x9 内角点"的情况
CANDIDATES = [(11, 8), (12, 9), (13, 10), (10, 7), (9, 6), (8, 5), (7, 5), (6, 4),
              (14, 11), (15, 12), (5, 4), (4, 3)]


def main() -> int:
    ap = argparse.ArgumentParser(description="连拍 + 自动找标定板（不需要 GUI）")
    ap.add_argument("--out", default="/tmp/hunt")
    ap.add_argument("--n", type=int, default=12, help="拍几张")
    ap.add_argument("--auto", action="store_true",
                    help="自动按间隔连拍（你只管匀速转相机），不用敲回车")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--square-mm", type=float, default=20.0)
    ap.add_argument("--exposure", type=float, default=50000)
    ap.add_argument("--gain", type=float, default=20.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.png"):
        p.unlink()

    cap = open_camera("hik")
    cap.configure(width=1624, height=1240, pixel_format="Mono8",
                  exposure_us=args.exposure, gain_db=args.gain, frame_rate=4)
    it = cap.frames(None)

    print("=" * 66)
    print("找板子：慢慢转动相机，把标定板扫过画面")
    print("=" * 66)
    if args.auto:
        print(f"自动模式：每 {args.interval:.1f}s 拍一张，共 {args.n} 张。请匀速转动相机。")
    else:
        print(f"手动模式：每转一个方向停一下、按回车拍一张，共 {args.n} 张。")

    paths = []
    exposure_bad = []
    try:
        for k in range(args.n):
            if not args.auto:
                try:
                    input(f"  [{k+1}/{args.n}] 摆好姿势后按回车...")
                except EOFError:
                    print("  (非交互环境，自动改为一秒一张)")
            f = next(it)
            p = out / f"{k:03d}.png"
            import cv2
            cv2.imwrite(str(p), f.image)
            paths.append(p)
            print(f"     已存 {p.name}  mean={f.image.mean():5.1f}")
            # 曝光判定比均值有用得多：2026-09-14 那次整场采集就是因为过曝报废，
            # 而当时只打印了 mean=225.5 —— 看起来像个正常数字，没人会起疑。
            _st = exposure_stats(f.image)
            if _st["verdict"] != "ok":
                print("     " + format_hint(_st, args.exposure))
                exposure_bad.append(_st["verdict"])
            if args.auto:
                time.sleep(args.interval)
    finally:
        cap.close()

    print("\n" + "=" * 66)
    print("分析")
    print("=" * 66)
    any_hit = False
    for p in paths:
        import cv2
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        hits = []
        for cols, rows in CANDIDATES:
            d = detect(img, BoardSpec("chessboard", (cols, rows),
                                      args.square_mm / 1000.0))
            if d.found:
                hits.append((cols, rows, len(d.image_points)))
        if hits:
            any_hit = True
            print(f"  ✅ {p.name}: " + "; ".join(
                f"内角点 {c}x{r}（{n} 点）" for c, r, n in hits))
            print(f"      → 这就是你的规格：--pattern {hits[0][0]}x{hits[0][1]}"
                  f" --square-mm {args.square_mm:.0f}")
        else:
            print(f"  ❌ {p.name}: 无")

    print()
    if any_hit:
        print("🎉 找到板子了 —— 用上面报告的 --pattern 参数重跑 capture_h7.py 即可。")
        if exposure_bad:
            print(f"   ⚠️  但有 {len(exposure_bad)} 帧曝光不合格"
                  f"（{exposure_bad[0]}）—— 正式采集前先按上面的建议调曝光。")
    else:
        print("❌ 这一组里一帧都没检出。请打开")
        print(f"     {out}/000.png")
        print("   看一眼相机到底在拍什么，确认：")
        # 曝光问题优先说 —— 它是"板子明明在画面里却检不出"的头号原因，
        # 而且调一个参数就能解决，比怀疑板子/距离/对焦省事得多。
        if exposure_bad:
            print(f"   ⚠️ **先解决曝光**：{len(exposure_bad)}/{len(paths)} 帧曝光不合格"
                  f"（{exposure_bad[0]}）。")
            print("      过曝会让白格与背景一起顶到 255，角点检测直接失效；")
            print(f"      把 --exposure 从 {args.exposure:.0f} µs 往下调再看。")
        print("   ① 板子真的在镜头前方（不是背面/侧面）；")
        print("   ② 板子占画面 1/3 以上（太远角点太小）；")
        print("   ③ 板子没被反光/阴影糊掉；")
        print("   ④ 方格尺寸量过了（--square-mm 只影响尺度，不影响能否检出）。")
    return 0 if any_hit else 1


if __name__ == "__main__":
    raise SystemExit(main())
