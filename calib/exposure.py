#!/usr/bin/env python3
"""曝光质量判定：把"图糊了/黑成一片"变成一句能照着调的提示。

为什么需要这个模块
==================================================================
2026-09-14 的第一次真机试采集**整场报废**，原因不是算法也不是接线，而是
**曝光严重过曝**：那一帧 `mean = 225.5 ~ 248.8`（255 满量程），标定板白格与
背景全糊成一片，`findChessboardCornersSB` 一个角点都找不到。

而当时的工具**只把均值打印出来，不做判断** ——
`hunt_board.py` 打的是 `mean=225.5`，看起来像个正常数字，没人会盯着它起疑。
真正有价值的信息是"**这个数字意味着什么、我该把曝光调成多少**"。

所以这个模块只做一件事：把亮度统计翻译成 **OK / 过曝 / 欠曝 + 具体建议**。

判据为什么这样定
------------------------------------------------------------------
不用"均值在某个区间"当唯一判据 —— 棋盘格本身是黑白各半，**均值天然居中**，
一张完全过曝的图（全白）均值反而是 255 而不是"看起来很亮"的 150。
所以主判据是**饱和像素占比**：

* ``sat_hi``：≥250 的像素比例。过曝时白格、纸面、背景一起顶到 255，
  这个比例会飙到 20% 以上。**它才是过曝的直接证据。**
* ``sat_lo``：≤5 的像素比例。欠曝/没开镜头盖时接近 1.0。
* ``contrast``：p99 − p1。太低说明整幅图没有可用的黑白对比，
  即使均值好看也检不出角点。

用法::

    from exposure import exposure_stats, format_hint
    st = exposure_stats(gray)
    print(format_hint(st))        # "⚠️ 过曝：16% 像素已饱和（>=250），把曝光降到 1/3 试试"
    if st["verdict"] != "ok":
        ...
"""
from __future__ import annotations

import numpy as np

# 阈值集中在这里，方便按相机调（工业相机与笔记本摄像头满量程相同，
# 但噪声地板不同，实测 5/250 这组对有噪声的 UVC 也够用）。
SAT_HI_LEVEL = 250        # 视为"已饱和"的灰度
SAT_LO_LEVEL = 5
SAT_HI_WARN = 0.15        # 饱和比例超过它就判过曝
MEAN_HI_WARN = 200.0      # 均值兜底（大面积均匀白）
MEAN_LO_WARN = 30.0       # 均值兜底（欠曝）
CONTRAST_MIN = 40.0       # p99 - p1 的下限


def exposure_stats(image) -> dict:
    """统计一帧的曝光质量。

    参数
    ----
    image : ndarray
        灰度或彩色图（彩色会先转灰度）。支持 uint8；其它 dtype 会先归一化到
        0~255，这样即使调用方传的是 float 图也不会静默算错。

    返回
    ----
    dict，含 ``mean`` / ``sat_hi`` / ``sat_lo`` / ``contrast`` / ``verdict``
    （``"ok"`` / ``"too_bright"`` / ``"too_dark"`` / ``"low_contrast"``）
    以及 ``suggest_exposure_factor``（建议的曝光倍数，1.0 表示不用改）。
    """
    a = np.asarray(image)
    if a.ndim == 3:
        # 用整数运算避免浮点转换的开销；权重是标准的 Rec.601 亮度权重
        a = a[..., :3] @ np.array([0.299, 0.587, 0.114])
    if a.dtype != np.uint8:
        a = np.asarray(a, dtype=np.float64)
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        a = np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo) * 255.0
        a = a.astype(np.uint8)
    if a.size == 0:
        return {"mean": float("nan"), "sat_hi": 0.0, "sat_lo": 0.0,
                "contrast": 0.0, "verdict": "too_dark",
                "suggest_exposure_factor": 1.0, "n_pixels": 0}

    mean = float(a.mean())
    sat_hi = float((a >= SAT_HI_LEVEL).mean())
    sat_lo = float((a <= SAT_LO_LEVEL).mean())
    p1, p99 = np.percentile(a, (1, 99))
    contrast = float(p99 - p1)

    if sat_hi > SAT_HI_WARN or mean > MEAN_HI_WARN:
        verdict = "too_bright"
    elif mean < MEAN_LO_WARN or sat_lo > 0.5:
        verdict = "too_dark"
    elif contrast < CONTRAST_MIN:
        verdict = "low_contrast"
    else:
        verdict = "ok"

    # 建议倍数：过曝按超出比例往回压，欠曝按缺口往上抬，都给个保守值。
    factor = 1.0
    if verdict == "too_bright":
        # 目标是把饱和比例压到 ~2%，经验上亮度降一半能显著改善
        factor = 0.5 if mean > 220 else 0.7
    elif verdict == "too_dark":
        factor = 2.0 if mean < 15 else 1.5

    return {"mean": mean, "sat_hi": sat_hi, "sat_lo": sat_lo,
            "contrast": contrast, "verdict": verdict,
            "suggest_exposure_factor": factor, "n_pixels": int(a.size)}


def format_hint(st: dict, exposure_us: float | None = None) -> str:
    """把 :func:`exposure_stats` 的结果变成一句可执行的中文提示。"""
    v = st.get("verdict", "ok")
    npx = st.get("n_pixels", 0)
    if npx == 0:
        return "❌ 空图"
    head = (f"mean={st['mean']:.1f} 饱和={st['sat_hi']*100:.1f}% "
            f"对比={st['contrast']:.0f}")

    if v == "ok":
        return f"✅ 曝光正常（{head}）"

    f = st.get("suggest_exposure_factor", 1.0)
    if exposure_us and f != 1.0:
        sug = f"把 --exposure 从 {exposure_us:.0f} 改成约 {exposure_us*f:.0f} µs"
    elif f != 1.0:
        sug = f"把曝光乘 {f:.2f}"
    else:
        sug = "曝光不用改"

    if v == "too_bright":
        return (f"❌ **过曝**：{st['sat_hi']*100:.1f}% 的像素已经饱和（≥{SAT_HI_LEVEL}），"
                f"白格和背景糊成一片 → {sug}。\n"
                f"   （{head}；标定板检不出来的最常见原因就是这个，"
                f"2026-09-14 那次整场采集就废在这里）")
    if v == "too_dark":
        return (f"❌ **欠曝**：{head} → {sug}；也确认镜头盖已取下、"
                f"光圈/增益不是最小。")
    return (f"⚠️ **对比度不足**：{head}（p99−p1 < {CONTRAST_MIN:.0f}）→ "
            f"即使亮度正常也检不出角点：{sug}，但要查的是"
            f"**镜头是否对焦 / 照明是否均匀 / 板子是否在画面里**。")


def is_ok(image) -> bool:
    """便利函数：这一帧的曝光能不能用。"""
    return exposure_stats(image)["verdict"] == "ok"


if __name__ == "__main__":   # 自测：合成几种典型曝光，验证判定不误报
    rng = np.random.default_rng(0)
    h, w = 240, 320
    board = np.zeros((h, w), np.uint8)
    board[:] = 180
    for i in range(0, h, 40):
        for j in range(0, w, 40):
            if (i // 40 + j // 40) % 2 == 0:
                board[i:i + 40, j:j + 40] = 40
    cases = {
        "正常板子（理想）": board,
        "过曝（+90 顶到 255）": np.clip(board.astype(int) + 90, 0, 255).astype(np.uint8),
        "全白（盖没开/曝光拉满）": np.full((h, w), 255, np.uint8),
        "欠曝（×0.1）": (board * 0.1).astype(np.uint8),
        "全黑": np.zeros((h, w), np.uint8),
        "均匀灰（无对比）": np.full((h, w), 128, np.uint8),
        "带噪声的正常图": np.clip(board + rng.normal(0, 3, board.shape), 0, 255).astype(np.uint8),
    }
    print("曝光判定自测")
    print("=" * 72)
    bad = 0
    expect = {"正常板子（理想）": "ok", "过曝（+90 顶到 255）": "too_bright",
              "全白（盖没开/曝光拉满）": "too_bright", "欠曝（×0.1）": "too_dark",
              "全黑": "too_dark", "均匀灰（无对比）": "low_contrast",
              "带噪声的正常图": "ok"}
    for name, img in cases.items():
        st = exposure_stats(img)
        got = st["verdict"]
        ok = got == expect[name]
        bad += not ok
        print(f"{'✅' if ok else '❌'} {name:<22} → {got:<13} "
              f"（期望 {expect[name]}）")
        print(f"     {format_hint(st, 15000).splitlines()[0]}")
    print("=" * 72)
    print("✅ 全部符合预期" if bad == 0 else f"❌ {bad} 项不符预期")
    raise SystemExit(1 if bad else 0)
