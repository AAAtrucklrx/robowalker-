#!/usr/bin/env python3
"""C4 · 验证与报告：把"标定结果是否合理、判断依据是什么"变成可复现的产物。

任务书第 4 节要求**至少一种**验证方式，本模块默认全做：

| 方式 | 函数 | 回答什么问题 |
|---|---|---|
| 重投影误差 | （C1 已产出） | 内参好不好 |
| 标定残差 | （C1 已产出） | 内参是否收敛 |
| 运动一致性 | ``motion_consistency`` | 外参对不对（相机与 IMU 对同一段运动的描述是否一致） |
| 多次独立标定对比 | ``cross_validate`` / ``bootstrap_extrinsics`` | 结果稳不稳、置信区间多宽 |
| 与成熟工具对比 | （ROS 2 ``camera-calibration``，见 doc/对标RM主流做法.md） | 有没有系统性偏差 |
| IMU 噪声参数 | ``allan_deviation`` | 配置文件里的噪声密度到底该填多少 |

**贯穿全篇的一个原则**：每个数字都要能回答"多少算好"和"我达标了没有"。
所以 ``VERDICTS`` 里给的是阈值，报告里会逐项打勾打叉，而不是只丢一堆数。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


# ══════════════════════════════════════════════════════════════
#  合格判据（写进报告，明确"多少算好"）
# ══════════════════════════════════════════════════════════════
VERDICTS = {
    "reprojection_rms_px": (0.5, 1.0, "像素", "重投影误差 RMS"),
    "motion_consistency_mean_deg": (1.0, 2.0, "度", "运动一致性均值"),
    "handeye_spread_deg": (0.5, 1.0, "度", "手眼多解法离散度"),
    "extrinsic_rot_std_deg": (1.0, 2.0, "度", "交叉验证旋转外参标准差"),
    "extrinsic_trans_std_m": (0.005, 0.02, "米", "交叉验证平移外参标准差"),
    "lever_arm_residual_ms2": (0.05, 0.1, "m/s²", "杠杆臂 LS 残差"),
    "gravity_norm_error": (0.05, 0.2, "m/s²", "重力模长偏差（内建自检）"),
}


def verdict(key: str, value: float) -> str:
    """返回 ✅ / ⚠️ / ❌ —— 好 / 可接受 / 不达标。"""
    if key not in VERDICTS or value is None or not np.isfinite(value):
        return "❔"
    good, ok, _, _ = VERDICTS[key]
    v = abs(float(value))
    return "✅" if v <= good else ("⚠️" if v <= ok else "❌")


def verdict_text(key: str) -> str:
    if key not in VERDICTS:
        return ""
    good, ok, unit, _ = VERDICTS[key]
    return f"好 < {good} {unit}，可接受 < {ok} {unit}"


# ══════════════════════════════════════════════════════════════
#  ① Allan 方差：IMU 噪声参数
# ══════════════════════════════════════════════════════════════
def allan_deviation(t: np.ndarray, x: np.ndarray, n_tau: int = 40):
    """重叠 Allan 偏差（IEEE Std 952）。

    参数
    ----
    t : (N,) 等间隔时间戳（秒）
    x : (N,) 或 (N,3) 数据（**必须是静止段**：Allan 方差只对静态数据有意义）

    返回 ``(taus, adev)``，``adev`` 形状与 ``x`` 的列数对应。

    怎么读这张图（报告里要写）：
    * 曲线在 ``tau`` 小的一段的斜率 **−1/2** → 角度/速度随机游走（白噪声），
      ``N = adev(tau) · sqrt(tau)`` 即噪声密度；
    * **+1/2** 斜率 → 速率随机游走；
    * 曲线**最低点**附近对应零偏不稳定性，``B ≈ adev_min / 0.664``。
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    n = len(t)
    if n < 32:
        raise ValueError(f"样本只有 {n} 个，Allan 方差至少要 32 个（越长越好，建议 ≥10 分钟）")
    dt = float(np.median(np.diff(t)))
    if np.std(np.diff(t)) > 0.1 * dt:
        raise ValueError("时间戳不等间隔，Allan 方差要求均匀采样")

    max_m = max(2, n // 4)
    ms = np.unique(np.round(np.logspace(np.log10(1), np.log10(max_m),
                                        n_tau)).astype(int))
    taus, adevs = [], []
    for m in ms:
        if m < 2 or n - 2 * m < 8:
            continue
        # 重叠式：对每个起点算 bin 均值
        csum = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)], axis=0)
        avgs = (csum[m:] - csum[:-m]) / m          # 长度 n-m+1
        d = avgs[m:] - avgs[:-m]                   # 长度 n-2m+1
        if len(d) < 4:
            continue
        adev = np.sqrt(np.sum(d ** 2, axis=0) / (2.0 * len(d)))
        taus.append(m * dt)
        adevs.append(adev)
    if not taus:
        raise ValueError("可用 tau 太少，数据长度不足")
    return np.asarray(taus), np.asarray(adevs)


def estimate_imu_noise(t: np.ndarray, gyro: np.ndarray, accel: np.ndarray,
                       tau_ref: float = 1.0) -> dict:
    """从 Allan 偏差估计陀螺/加速度计的噪声密度与零偏不稳定性。

    ``tau_ref`` 处取噪声密度（白噪声段），并取曲线最低点估零偏不稳定性。
    这些都是**要填进 config 的 `imu` 段**的值。
    """
    out = {"tau_ref_s": tau_ref}
    for name, data, unit in (("gyro", gyro, "rad/s"), ("accel", accel, "m/s^2")):
        try:
            taus, adev = allan_deviation(t, data)
        except ValueError as e:
            out[name] = {"error": str(e)}
            continue
        # 噪声密度：在最接近 tau_ref 的点取 adev*sqrt(tau)
        i = int(np.argmin(np.abs(taus - tau_ref)))
        n_density = adev[i] * np.sqrt(taus[i])
        # 零偏不稳定性：曲线最低点（取三轴平均曲线的最低点所在的那个 tau，
        # 然后取该 tau 上三个轴各自的值 —— 不要用标量，否则下面展开会报错）
        j = int(np.argmin(adev.min(axis=1)))
        bias_instab = adev[j] / 0.664
        out[name] = {
            "noise_density": [float(v) for v in n_density],
            "noise_density_unit": f"{unit}/sqrt(Hz)",
            "bias_instability": [float(v) for v in bias_instab],
            "bias_instability_unit": unit,
            "adev_min": float(adev.min()),
            "tau_at_min_s": float(taus[j]),
            "taus_s": taus.tolist(),
            "adev": adev.tolist(),
        }
    return out


# ══════════════════════════════════════════════════════════════
#  ② 交叉验证 / Bootstrap：多次独立标定对比
# ══════════════════════════════════════════════════════════════
def _rot_spread_deg(Rs: list) -> float:
    """一组旋转矩阵相对其"平均值"的散布（度）。"""
    if len(Rs) < 2:
        return float("nan")
    import cv2

    rvs = np.array([cv2.Rodrigues(R)[0].reshape(3) for R in Rs])
    R_mean, _ = cv2.Rodrigues(np.mean(rvs, axis=0).reshape(3, 1))
    return float(np.rad2deg(np.mean([
        np.arccos(np.clip((np.trace(R_mean.T @ R) - 1) / 2, -1, 1)) for R in Rs
    ])))


def cross_validate(frames: list, k: int, solve_fn, seed: int = 0) -> dict:
    """k 折交叉验证：把帧切成 k 份，每份单独标一次，比外参的离散度。

    ``frames``：任意可切片的帧列表（如 (t, R_CB, t_CB) 的索引）
    ``solve_fn``：``solve_fn(subset) -> (R_IC, t_IC)``

    离散度小 = 结果稳定、不依赖特定数据；离散度大 = 数据量不够或有系统误差。
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(frames))
    folds = np.array_split(idx, k)
    Rs, ts = [], []
    for f in folds:
        if len(f) < 20:
            continue
        try:
            R, tt = solve_fn([frames[i] for i in f])
            Rs.append(R)
            ts.append(np.asarray(tt).reshape(3))
        except Exception:  # noqa: BLE001
            continue
    if len(Rs) < 2:
        return {"n_folds": len(Rs), "error": "有效折数不足，无法交叉验证"}
    ts = np.asarray(ts)
    return {
        "n_folds": len(Rs),
        "rot_std_deg": _rot_spread_deg(Rs),
        "trans_std_m": float(np.linalg.norm(ts.std(axis=0))),
        "trans_mean_m": [float(v) for v in ts.mean(axis=0)],
        "per_fold_R": Rs,
        "per_fold_t": ts.tolist(),
    }


def bootstrap_extrinsics(frames: list, solve_fn, n_boot: int = 20,
                         seed: int = 0, frac: float = 0.7) -> dict:
    """Bootstrap：有放回重采样，得到外参的经验分布与置信区间。

    比交叉验证更适合"数据不多"的场景。返回 95% 区间（用 2.5/97.5 分位）。
    """
    rng = np.random.default_rng(seed)
    n = len(frames)
    m = max(20, int(n * frac))
    Rs, ts = [], []
    for _ in range(n_boot):
        pick = [frames[i] for i in rng.integers(0, n, m)]
        try:
            R, tt = solve_fn(pick)
            Rs.append(R)
            ts.append(np.asarray(tt).reshape(3))
        except Exception:  # noqa: BLE001
            continue
    if len(Rs) < 5:
        return {"n_boot": len(Rs), "error": "有效重采样次数不足"}
    ts = np.asarray(ts)
    return {
        "n_boot": len(Rs),
        "trans_mean_m": [float(v) for v in ts.mean(axis=0)],
        "trans_ci95_lo_m": [float(np.percentile(ts[:, j], 2.5)) for j in range(3)],
        "trans_ci95_hi_m": [float(np.percentile(ts[:, j], 97.5)) for j in range(3)],
        "trans_std_m": float(np.linalg.norm(ts.std(axis=0))),
        "rot_spread_deg": _rot_spread_deg(Rs),
        "rot_samples": Rs,
        "trans_samples": ts.tolist(),
    }


# ══════════════════════════════════════════════════════════════
#  ③ 报告生成
# ══════════════════════════════════════════════════════════════
def _row(name, value, key=None, unit="", note=""):
    v = "n/a" if value is None or (isinstance(value, float) and not np.isfinite(value)) \
        else f"{value:.4f}"
    mark = verdict(key, value) if key else ""
    vt = f"（{verdict_text(key)}）" if key and key in VERDICTS else ""
    return f"| {name} | {v} {unit} | {mark} {vt} | {note} |"


def build_report(intr: dict, extr: dict, val: dict, meta: dict,
                 imu_noise: dict | None = None) -> str:
    """生成 Markdown 验证报告。**每个指标都带合格判据**，而不只是丢数字。"""
    L = []
    A = L.append
    A("# Camera-IMU 联合标定 · 验证报告\n")
    A(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    A(f"> 数据：`{meta.get('session', 'n/a')}`"
      f"（{meta.get('n_images', '?')} 帧图像 / {meta.get('n_imu', '?')} 条 IMU）  ")
    A("> 坐标系约定：**T_IC 表示 相机系 C → IMU 系 I**，p_I = R_IC·p_C + t_IC\n")

    A("## 1. 结论一句话\n")
    A(meta.get("verdict_line", "（见下表逐项判断）"))
    A("")

    A("## 2. 相机内参\n")
    i = (intr or {}).get("intrinsics", {})
    A("| 参数 | 值 | 判定 | 说明 |")
    A("|---|---|---|---|")
    if i:
        A(f"| fx | {i.get('fx', float('nan')):.3f} px | | |")
        A(f"| fy | {i.get('fy', float('nan')):.3f} px | | |")
        A(f"| cx | {i.get('cx', float('nan')):.3f} px | | |")
        A(f"| cy | {i.get('cy', float('nan')):.3f} px | | |")
    d = (intr or {}).get("distortion", {}).get("coefficients", [])
    if d:
        A(f"| 畸变 [k1 k2 p1 p2 k3] | {np.array2string(np.asarray(d), precision=5)} | | |")
    A("")
    A("| 指标 | 值 | 判定 | 说明 |")
    A("|---|---|---|---|")
    A(_row("标定残差 RMS (train)", (intr or {}).get("reprojection_error_rms"),
           "reprojection_rms_px", "px", "用标定数据算，天然偏乐观"))
    vv = (val or {}).get("intrinsics_val", {})
    A(_row("**重投影误差 RMS (val)**", vv.get("rms_px"),
           "reprojection_rms_px", "px", "**没参与标定的图，才是诚实指标**"))
    A(_row("重投影误差 max (val)", vv.get("max_px"), None, "px", f"{vv.get('n', 0)} 张验证图"))
    A("")

    A("## 3. Camera-IMU 外参\n")
    e = (extr or {})
    T = e.get("T_IC")
    if T:
        A("```")
        A("T_IC（相机系 → IMU 系）=")
        for row in T:
            A("  [" + "  ".join(f"{v: .6f}" for v in row) + "]")
        A("```")
        A(f"- 旋转（ZYX 欧拉角）：{np.array2string(np.asarray(e.get('rotation_euler_deg_zyx', [])), precision=3)} °")
        A(f"- 平移：{np.array2string(np.asarray(e.get('translation_m', [])), precision=5)} m"
          f"（|t| = {e.get('translation_norm_m', float('nan'))*100:.3f} cm）")
        A(f"- 杠杆臂（IMU 原点在相机系下）："
          f"{np.array2string(np.asarray(e.get('lever_arm_in_camera_m', [])), precision=5)} m")
    A("")

    A("## 4. 验证指标\n")
    A("| 指标 | 值 | 判定 | 说明 |")
    A("|---|---|---|---|")
    mc = (val or {}).get("motion_consistency", {})
    A(_row("运动一致性（均值）", mc.get("mean_deg"), "motion_consistency_mean_deg",
           "度", "相机与 IMU 对同一段运动的描述差角"))
    A(_row("运动一致性（max）", mc.get("max_deg"), None, "度", f"{mc.get('n', 0)} 对区间"))
    A(_row("手眼多解法离散度", (val or {}).get("handeye_spread_deg"),
           "handeye_spread_deg", "度", "多种解法互相差多少"))
    cv = (val or {}).get("cross_validation", {})
    A(_row("交叉验证 旋转标准差", cv.get("rot_std_deg"),
           "extrinsic_rot_std_deg", "度", f"{cv.get('n_folds', 0)} 折"))
    A(_row("交叉验证 平移标准差", cv.get("trans_std_m"),
           "extrinsic_trans_std_m", "米", f"{cv.get('n_folds', 0)} 折"))
    lev = (val or {}).get("lever_arm", {})
    A(_row("杠杆臂 LS 残差", lev.get("residual_rms_ms2"),
           "lever_arm_residual_ms2", "m/s²", "应明显小于杠杆臂信号量级(~0.06)"))
    A(_row("重力模长偏差", lev.get("gravity_norm_error"),
           "gravity_norm_error", "m/s²", "**内建自检**：|g| 应等于 9.80665"))
    A(_row("时间偏移 τ", (val or {}).get("time_offset", {}).get("tau_s"),
           None, "s", "加到 IMU 时间戳上即得主机时钟"))
    A("")

    bs = (val or {}).get("bootstrap", {})
    if bs and "error" not in bs:
        A("### Bootstrap 置信区间（平移外参）\n")
        A(f"重采样 {bs.get('n_boot')} 次：")
        for j, ax in enumerate("xyz"):
            lo = bs.get("trans_ci95_lo_m", [np.nan]*3)[j]
            hi = bs.get("trans_ci95_hi_m", [np.nan]*3)[j]
            A(f"- t_{ax} = {bs['trans_mean_m'][j]:+.5f} m，95% CI [{lo:+.5f}, {hi:+.5f}]")
        A("")

    if imu_noise:
        A("## 5. IMU 噪声参数（Allan 方差）\n")
        A("> 建议填入 `config/calibration.yaml` 的 `imu` 段。"
          "注意：Allan 方差只对**静止段**数据有意义。\n")
        A("| 通道 | 噪声密度 | 零偏不稳定性 | 曲线最低点 tau |")
        A("|---|---|---|---|")
        for name in ("gyro", "accel"):
            v = imu_noise.get(name, {})
            if "error" in v:
                A(f"| {name} | 数据不足：{v['error']} | | |")
                continue
            nd = np.linalg.norm(v.get("noise_density", [np.nan]))
            bi = np.linalg.norm(v.get("bias_instability", [np.nan]))
            A(f"| {name} | {nd:.6g} {v.get('noise_density_unit','')} | "
              f"{bi:.6g} {v.get('bias_instability_unit','')} | "
              f"{v.get('tau_at_min_s', float('nan')):.2f} s |")
        A("")

    A("## 6. 判断依据（回答「结果是否合理」）\n")
    A("1. **重投影误差**在验证集上达标 → 内参可信，且不是用标定数据自证的；")
    A("2. **运动一致性**：由 AX=XB 的**相对旋转**残差给出，与坐标系朝向无关")
    A("   （陀螺没有绝对 yaw 基准，用绝对姿态比会凭空多出几十度误差）；")
    A("3. **交叉验证 / Bootstrap**：多份独立子集标出来的外参离散度小 → 结果不依赖特定数据；")
    A("4. **|g| 内建自检**：联合估计出的重力模长应等于 9.80665。"
      "偏离说明板系与世界系未对齐，或数据有问题；")
    A("5. **杠杆臂 LS 残差**应明显小于杠杆臂信号量级（约 0.06 m/s²），否则平移不可信。")
    A("")
    if meta.get("notes"):
        A("## 7. 补充说明\n")
        for n in meta["notes"]:
            A(f"- {n}")
        A("")
    return "\n".join(L)


def write_report(text: str, out_dir: str | Path) -> Path:
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    p = out / "report.md"
    p.write_text(text, encoding="utf-8")
    return p
