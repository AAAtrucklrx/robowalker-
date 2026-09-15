#!/usr/bin/env python3
"""C2 · 相机-IMU 时间对齐（估计两个时钟之间的常量偏移 τ）。

问题
------------------------------------------------------------------
相机时间戳来自主机单调时钟，IMU 时间戳来自板载毫秒计数器（或另一个时钟）。
两者之间有一个未知常量偏移 τ：``t_host = t_imu + tau``。

**为什么这件事比算法本身更容易毁掉结果**：假设云台以 90 °/s 转动，
时间戳差 10 ms → 姿态误差 ``90 × 0.01 = 0.9°``。而验收标准要求旋转
外参误差 < 1~2°。所以 τ 不估准，后面全白做。

估计方法：**转动角度匹配**（不依赖外参）
------------------------------------------------------------------
相邻两帧相机之间：

* 相机侧：由 PnP 位姿得到相对旋转 ``ΔR_cam = R_CB(t_{k+1}) R_CB(t_k)^T``，
  取其旋转角 ``θ_cam = |log(ΔR_cam)|``；
* IMU 侧：把陀螺在对应时间窗内积分，取模长 ``θ_imu = |∫ω dt|``。

关键点：因为 ``R_IC ΔR_cam = ΔR_imu R_IC``，**两个相对旋转的旋转角必然相等**
（相似矩阵的迹相同）。所以用"角度"做匹配就**不需要预先知道外参 R_IC**——
可以先把 τ 解出来，再去解外参。这个顺序很重要。

代价函数：``Σ_k (θ_cam,k − θ_imu,k(τ))²``，在一维上网格搜索 + 抛物线插值细化。

用法::

    .venv/bin/python calib/sync.py --session data/session_01
"""
from __future__ import annotations

import numpy as np


# ══════════════════════════════════════════════════════════════
def rot_angle(R: np.ndarray) -> float:
    """旋转矩阵的旋转角（弧度）。用迹公式，比 log 稳。"""
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def camera_relative_angles(cam_times: np.ndarray, cam_R: np.ndarray):
    """相邻相机帧之间的旋转角。

    参数
    ----
    cam_R : (N,3,3)，``R_CB``（板系 -> 相机系），从 PnP 得到。

    返回 ``(t_a, t_b, angles)``：每个区间的起止时刻与旋转角。
    """
    t = np.asarray(cam_times, float)
    R = np.asarray(cam_R, float)
    n = len(t)
    if n < 2:
        raise ValueError("至少需要 2 帧")
    angles = np.array([rot_angle(R[i + 1] @ R[i].T) for i in range(n - 1)])
    return t[:-1], t[1:], angles


GYRO_RESAMPLE_HZ = 10_000.0


def prepare_gyro_integral(t_imu: np.ndarray, gyro: np.ndarray,
                          rate: float = GYRO_RESAMPLE_HZ):
    """把陀螺重采样到均匀网格并做**累积梯形积分**，返回 ``(t_u, W)``。

    为什么要多这一步：如果直接对"落在窗口内的原始样本"积分，窗口边界会吸附到
    1 kHz 的样本点上，代价函数随 τ 变成**阶梯状**，抛物线细化就完全失效了。
    重采样到 10 kHz 后，τ 的量化误差降到 0.1 ms —— 90 °/s 下只对应 0.009°，
    可以忽略。

    ``W[i] = ∫_{t_u[0]}^{t_u[i]} ω dt``（逐分量）。
    """
    t_imu = np.asarray(t_imu, float)
    w = np.asarray(gyro, float)
    if len(t_imu) < 2:
        raise ValueError("IMU 样本太少")
    dt = 1.0 / rate
    t_u = np.arange(t_imu[0], t_imu[-1], dt)
    w_u = np.stack([np.interp(t_u, t_imu, w[:, j])
                    for j in range(w.shape[1])], axis=1)
    W = np.zeros((len(t_u), w.shape[1]))
    W[1:] = np.cumsum(0.5 * (w_u[:-1] + w_u[1:]) * dt, axis=0)
    return t_u, W


def integrated_angles(t_u: np.ndarray, W: np.ndarray, a, b) -> np.ndarray:
    """窗口 ``[a,b]``（IMU 时间轴）内陀螺积分的模长 ``|∫ω dt|``。a/b 可为数组。"""
    a = np.atleast_1d(np.asarray(a, float))
    b = np.atleast_1d(np.asarray(b, float))
    n = len(t_u)
    ia = np.clip(np.searchsorted(t_u, a, side="left"), 0, n - 1)
    ib = np.clip(np.searchsorted(t_u, b, side="right"), 0, n - 1)
    d = W[ib] - W[ia]
    d[ib <= ia] = 0.0
    return np.linalg.norm(d, axis=1)


def offset_cost(t_u, W, t_a, t_b, cam_angles, tau) -> float:
    """给定 τ 的代价：相机与 IMU 在同一时间窗内的转动角度差平方和。"""
    imu_ang = integrated_angles(t_u, W, np.asarray(t_a) - tau, np.asarray(t_b) - tau)
    return float(np.mean((imu_ang - cam_angles) ** 2))


def estimate_offset(cam_times: np.ndarray, cam_R: np.ndarray,
                    t_imu: np.ndarray, gyro: np.ndarray,
                    search_range: float = 0.5, coarse_step: float = 2e-3,
                    min_angle_rad: float = 0.01,
                    verbose: bool = False) -> dict:
    """估计时钟偏移 τ，使 ``t_host = t_imu + tau``。

    只使用转动**角度**，因此不依赖外参 R_IC。

    参数
    ----
    cam_times : (N,) 相机时间戳（主机时钟，秒）
    cam_R     : (N,3,3) 每帧的 R_CB（板系->相机系）
    t_imu     : (M,) IMU 时间戳（IMU 自己的时钟）
    gyro      : (M,3) 角速度 rad/s
    search_range : 在 ±该秒数内搜索
    min_angle_rad : 转动太小的区间丢弃（信噪比太低）

    返回 dict：tau、代价曲线的对比度、残差、以及网格数据（用于画图/报告）。
    """
    cam_times = np.asarray(cam_times, float)
    t_imu = np.asarray(t_imu, float)
    gyro = np.asarray(gyro, float)

    # 相机时间戳必须单调
    if np.any(np.diff(cam_times) <= 0):
        raise ValueError("相机时间戳不是严格单调的")

    t_a, t_b, cam_ang = camera_relative_angles(cam_times, cam_R)
    keep = cam_ang >= min_angle_rad
    if keep.sum() < 3:
        raise ValueError(
            f"转动幅度足够的区间只有 {int(keep.sum())} 个。"
            f"采集时需要更明显地转动（或降低 min_angle_rad）")
    t_a, t_b, cam_ang = t_a[keep], t_b[keep], cam_ang[keep]
    t_mid = 0.5 * (t_a + t_b)

    # 陀螺重采样 + 累积积分，只做一次
    t_u, W = prepare_gyro_integral(t_imu, gyro)

    # 用一个粗网格先把 τ 定位到 coarse_step 以内
    taus = np.arange(-search_range, search_range + 1e-12, coarse_step)
    costs = np.array([
        offset_cost(t_u, W, t_a, t_b, cam_ang, tau) for tau in taus
    ])

    i_best = int(np.argmin(costs))
    tau0 = float(taus[i_best])

    # 抛物线插值细化（用最优点的左右邻居拟合二次曲线）
    if 0 < i_best < len(taus) - 1:
        y0, y1, y2 = costs[i_best - 1], costs[i_best], costs[i_best + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-18:
            delta = 0.5 * (y0 - y2) / denom
            delta = float(np.clip(delta, -1.0, 1.0))
            tau0 = tau0 + delta * coarse_step

    # 最优 τ 下的残差（角度）
    imu_ang = integrated_angles(t_u, W, t_a - tau0, t_b - tau0)
    resid = imu_ang - cam_ang
    rms_deg = float(np.rad2deg(np.sqrt(np.mean(resid ** 2))))

    # 代价曲线的对比度：最优点比中位数低多少倍 —— 用来判断 τ 是否可信
    contrast = float(np.median(costs) / max(costs[i_best], 1e-18))

    info = {
        "tau": tau0,
        "tau_coarse": float(taus[i_best]),
        "grid_taus": taus.tolist(),
        "grid_costs": costs.tolist(),
        "n_intervals": int(keep.sum()),
        "n_intervals_total": int(len(cam_ang) + (~keep).sum()),
        "residual_deg_rms": rms_deg,
        "residual_deg_max": float(np.rad2deg(np.abs(resid).max())),
        "cost_contrast": contrast,
        "mean_camera_angle_deg": float(np.rad2deg(cam_ang.mean())),
        "reliable": bool(rms_deg < 1.0 and contrast > 3.0),
    }

    if verbose:
        print(f"  τ = {tau0*1000:+.3f} ms   "
              f"（粗网格 {info['tau_coarse']*1000:+.3f} ms）")
        print(f"  参与匹配的区间 {info['n_intervals']} 个，"
              f"平均转动 {info['mean_camera_angle_deg']:.2f}°/帧")
        print(f"  角度残差 RMS = {rms_deg:.4f}°，max = {info['residual_deg_max']:.4f}°")
        print(f"  代价曲线对比度 = {contrast:.2f}x "
              f"({'可信' if contrast > 3 else '⚠️ 不够尖锐，τ 可能不可信'})")

    return info


# ══════════════════════════════════════════════════════════════
#  IMU 预处理（C3 也要用）
# ══════════════════════════════════════════════════════════════
def _despike(x: np.ndarray, width: int = 7) -> np.ndarray:
    """中位滤波去掉**孤立尖峰**（不改动其余样本）。

    为什么必须做：实测真机数据里，静止段每秒都会出现若干个**单帧**尖峰
    （例如连续值 ``[1.44 1.45 15.33 1.79 1.89]`` —— 孤立一帧跳到 15.3 °/s）。
    这类尖峰把 0.2 s 滚动窗口的标准差撑爆，于是**整段真正的静止被判成运动**，
    最终静止段 0 样本、零偏标定直接放弃。
    实测（session_01 前 12 s）：原实现 0 帧；中位滤波(7) 后恢复 **6901 帧
    = 6.9 s**。中位滤波只影响"判静止"这一步，不动原始数据。
    """
    from scipy.signal import medfilt

    x = np.asarray(x, float)
    if x.ndim == 1:
        return medfilt(x, width)
    return np.stack([medfilt(x[:, k], width) for k in range(x.shape[1])], axis=1)


def static_segment(t: np.ndarray, accel: np.ndarray, gyro: np.ndarray,
                   accel_std_max: float = 0.05, gyro_std_max: float = 0.02,
                   min_duration_s: float = 1.0,
                   window_s: float = 0.2,
                   despike: bool = True,
                   despike_width: int = 7) -> np.ndarray:
    """找出"完全静止"的样本（布尔掩码）。

    静止段是零偏标定与重力对齐的**唯一**可靠来源，所以采集时开头务必静置 2 秒。

    ⚠️ 这个函数踩过一次大坑：**最初按"样本数"定窗口**（25 个样本），在 1 kHz
    数据上窗口只有 25 ms，平滑运动中的一小段也很容易标准差低于阈值 ——
    结果把**整段运动数据判成了静止**（6999/7000 样本），于是减掉一个凭空的
    陀螺零偏，后面 R_WI 积分、时间对齐、外参**全部悄悄错掉且不报错**。

    现在改成三条硬性约束：

    1. 窗口按**秒**算（默认 0.2 s），不同采样率下行为一致；
    2. 阈值收紧（默认 0.05 m/s² / 0.02 rad/s）；
    3. **只保留连续时长 ≥ ``min_duration_s`` 的段** —— 零散的短窗不算静止。

    ⚠️ 后来又踩了第二个坑：真机 IMU 流里每秒都有**孤立单帧尖峰**
    （见 :func:`_despike`），滚动标准差被撑爆 → **真静止被判成运动**，
    于是"静止段 0 样本"。现在判静止前先中位滤波去尖峰（``despike=True``），
    实测把 0 帧恢复成 6.9 s。
    """
    t = np.asarray(t, float)
    a = np.asarray(accel, float)
    g = np.asarray(gyro, float)
    if despike:
        # 只在**判静止**时用去尖峰版本；返回的掩码索引对应原始数据，
        # 零偏仍从原始数据取（但用稳健统计量，见 estimate_bias）。
        a = _despike(a, despike_width)
        g = _despike(g, despike_width)
    n = len(t)
    if n < 16:
        return np.zeros(n, bool)
    dt = float(np.median(np.diff(t)))
    win = max(11, int(round(window_s / max(dt, 1e-9))))

    ok = np.zeros(n, bool)
    for i in range(n - win):
        if a[i:i + win].std(axis=0).max() < accel_std_max and \
           g[i:i + win].std(axis=0).max() < gyro_std_max:
            ok[i:i + win] = True

    # 只保留足够长的连续段
    need = max(1, int(round(min_duration_s / max(dt, 1e-9))))
    out = np.zeros(n, bool)
    i = 0
    while i < n:
        if ok[i]:
            j = i
            while j < n and ok[j]:
                j += 1
            if j - i >= need:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def estimate_bias(t: np.ndarray, accel: np.ndarray, gyro: np.ndarray,
                  mask: np.ndarray | None = None,
                  accel_std_max: float = 0.05, gyro_std_max: float = 0.02,
                  min_duration_s: float = 1.0) -> dict:
    """从静止段估计陀螺零偏与加速度零偏。

    陀螺零偏直接就是静止段的均值。加速度零偏采用"先定重力方向，再取残差"：
    用静止段加速度均值确定 IMU 系下重力方向，零偏 = 均值 − 该方向 × 9.80665。

    **带一道安全阀**：若"静止段"占了数据的 50% 以上就判定失败并给出原因 ——
    标定采集本来就是让人动的，整段都静止说明阈值把运动数据也吞了。
    宁可明确报错（退回不减零偏），也不要悄悄减掉一个错误的零偏。
    """
    t = np.asarray(t, float)
    a = np.asarray(accel, float)
    g = np.asarray(gyro, float)
    if mask is None:
        mask = static_segment(t, a, g, accel_std_max, gyro_std_max,
                              min_duration_s)
    n_static = int(np.sum(mask))
    frac = n_static / max(len(a), 1)
    if n_static < 200:
        return {"ok": False, "n_static": n_static, "static_fraction": frac,
                "reason": f"静止段只有 {n_static} 个样本（<200）。"
                          f"采集时开头请完全静置 2 秒以上"}
    if frac > 0.5:
        return {"ok": False, "n_static": n_static, "static_fraction": frac,
                "reason": f"静止段占了 {frac*100:.1f}% 的数据 —— 标定采集本来就"
                          f"要让人动起来，这不合理，说明静止判据把运动数据也吞了。"
                          f"已放弃零偏标定（不减零偏比减错零偏差）"}
    # 用**中位数**而不是均值取零偏：均值会被残留的孤立尖峰拉偏，而零偏是要
    # 从后面每一帧里减掉的量，偏一点就整体污染。中位数对少量离群点免疫。
    a_m = np.median(a[mask], axis=0)
    g_m = np.median(g[mask], axis=0)
    g_dir = a_m / np.linalg.norm(a_m)
    accel_bias = a_m - g_dir * 9.80665
    return {
        "ok": True,
        "n_static": n_static,
        "static_fraction": frac,
        "static_mask": mask,
        "gyro_bias": g_m,
        "accel_bias": accel_bias,
        "gravity_dir_in_imu": g_dir,
        "accel_norm_mean": float(np.linalg.norm(a_m)),
    }


def integrate_gyro(t: np.ndarray, gyro: np.ndarray,
                   R0: np.ndarray | None = None) -> np.ndarray:
    """用陀螺积分姿态，返回 ``R_WI(t)`` 序列 (N,3,3)。

    零阶保持（每步用区间起点的角速度）在 1 kHz 下足够准。
    没有磁力计 → **yaw 不可观测**，长时间积分必然漂移，所以只适合
    短时间窗（标定场景足够）。
    """
    import cv2

    t = np.asarray(t, float)
    w = np.asarray(gyro, float)
    n = len(t)
    R = np.eye(3) if R0 is None else np.asarray(R0, float).copy()
    out = np.zeros((n, 3, 3))
    out[0] = R
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        rv = w[i - 1] * dt
        dR, _ = cv2.Rodrigues(rv.reshape(3, 1))
        R = R @ dR
        out[i] = R
    return out


def resample_to(t_src: np.ndarray, y_src: np.ndarray,
                t_dst: np.ndarray) -> np.ndarray:
    """线性插值重采样（逐列）。"""
    t_src = np.asarray(t_src, float)
    y_src = np.asarray(y_src, float)
    if y_src.ndim == 1:
        return np.interp(t_dst, t_src, y_src)
    cols = [np.interp(t_dst, t_src, y_src[:, j]) for j in range(y_src.shape[1])]
    return np.stack(cols, axis=1)


# ══════════════════════════════════════════════════════════════
def main() -> int:
    import argparse
    import sys
    from pathlib import Path

    import cv2

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from intrinsics import Intrinsics
    from io_data import load_session
    from target_detect import BoardSpec, detect

    ap = argparse.ArgumentParser(description="C2 相机-IMU 时间对齐")
    ap.add_argument("--session", required=True, help="采集目录（含 images/ 与 imu.txt）")
    ap.add_argument("--pattern", default="9x6")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--intrinsics", default=None, help="内参 yaml；不给则用初始值")
    ap.add_argument("--search", type=float, default=0.5, help="±搜索范围（秒）")
    ap.add_argument("--step", type=float, default=2e-3, help="粗网格步长（秒）")
    args = ap.parse_args()

    c, r = (int(v) for v in args.pattern.lower().split("x"))
    spec = BoardSpec("chessboard", (c, r), args.square_mm / 1000.0)
    sess = load_session(args.session)
    print(f"装载 {sess.root}")
    for k, v in sess.summary().items():
        print(f"  {k:16s} {v}")

    if sess.imu is None:
        print("❌ 没有 IMU 数据")
        return 2

    intr = Intrinsics(np.array([[600., 0, 640], [0, 600., 480], [0, 0, 1]]),
                      np.zeros(5), (1280, 960))
    if args.intrinsics:
        import yaml
        intr = Intrinsics.from_dict(yaml.safe_load(Path(args.intrinsics).read_text()))

    # 逐帧 PnP 得到 R_CB
    Rs, Ts, ts = [], [], []
    for p, t in zip(sess.image_paths, sess.image_ts):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        d = detect(gray, spec)
        if not d.found:
            continue
        ok, rvec, tvec = cv2.solvePnP(d.object_points, d.image_points,
                                      intr.K, intr.dist)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        Rs.append(R); Ts.append(tvec.reshape(3)); ts.append(t)
    print(f"\nPnP 成功 {len(Rs)}/{len(sess.image_paths)} 帧")

    info = estimate_offset(np.array(ts), np.array(Rs), sess.imu.t,
                           sess.imu.gyro, search_range=args.search,
                           coarse_step=args.step, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
