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
def static_segment(accel: np.ndarray, gyro: np.ndarray,
                   accel_std_max: float = 0.08, gyro_std_max: float = 0.03,
                   min_len: int = 100) -> np.ndarray:
    """找出"完全静止"的样本（布尔掩码）。

    静止段是零偏标定与重力对齐的唯一可靠来源，所以采集时开头务必静置 2 秒。
    """
    a = np.asarray(accel, float)
    g = np.asarray(gyro, float)
    # 滑动窗口标准差
    win = max(11, min_len // 4)
    ok = np.zeros(len(a), bool)
    for i in range(len(a) - win):
        if a[i:i + win].std(axis=0).max() < accel_std_max and \
           g[i:i + win].std(axis=0).max() < gyro_std_max:
            ok[i:i + win] = True
    return ok


def estimate_bias(accel: np.ndarray, gyro: np.ndarray,
                  mask: np.ndarray | None = None) -> dict:
    """从静止段估计陀螺零偏与加速度零偏。

    陀螺零偏直接就是静止段的均值。加速度零偏需要先知道重力方向与大小，
    这里采用"先做重力对齐，再取残差"的做法：
    用静止段加速度均值确定 IMU 系下重力的方向，零偏 = 均值 - 该方向上的 9.80665。
    """
    a = np.asarray(accel, float)
    g = np.asarray(gyro, float)
    if mask is None:
        mask = static_segment(a, g)
    if mask.sum() < 20:
        return {"ok": False, "n_static": int(mask.sum()),
                "reason": "找不到足够的静止段（采集时开头请静置 2 秒）"}
    a_m = a[mask].mean(axis=0)
    g_m = g[mask].mean(axis=0)
    g_dir = a_m / np.linalg.norm(a_m)
    accel_bias = a_m - g_dir * 9.80665
    return {
        "ok": True,
        "n_static": int(mask.sum()),
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
