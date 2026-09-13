#!/usr/bin/env python3
"""C3 · Camera-IMU 外参 T_IC（旋转 + 平移）。

**约定（本文件所有符号都按这个来，写进报告时请照抄）**

    T_IC = [[R_IC, t_IC], [0, 1]]，表示 **相机系 C → IMU 系 I**：
        p_I = R_IC · p_C + t_IC

    杠杆臂 r ≡ IMU 原点在相机系下的坐标 = t_CI = -R_IC^T · t_IC
    反过来：t_IC = -R_IC · r

旋转：手眼标定 AX = XB
------------------------------------------------------------------
由"标定板固定不动 + 相机与 IMU 刚性固连"可得

    A X = X B,
    A = (T_g^b(2))^{-1} T_g^b(1)   ,  B = T_t^c(2) (T_t^c(1))^{-1},  X = T_c^g

映射到我们的物理量（g=IMU, c=相机, t=标定板, b=世界=标定板）：

    T_g^b = T_I^W  → **旋转就是 R_WI**（IMU 姿态，由陀螺积分得到）
    T_t^c = T_B^C  → PnP 直接给出的 R_CB
    X     = T_C^I  → 正是我们要的 T_IC

⚠️ **踩过的坑**：``R_gripper2base`` 要传 ``R_WI`` 本身，传 ``R_WI.T`` 会让
所有求解器一致地错几十度，而且**不报错**。本仓用真值数据把这个约定卡过一遍
（见 ``tools/selftest_pipeline.py`` 第 ② 项）。

只用陀螺、不需要加速度，也不需要 IMU 的绝对位置 —— 这是旋转部分最舒服的地方。

平移：只能用加速度计的杠杆臂效应
------------------------------------------------------------------
坑在于：**6 轴 IMU 测不到自己的绝对位置**，所以 ``T_g^b`` 的平移部分拿不到，
``AX=XB`` 的平移方程用不了。业界通用做法是用加速度计的**杠杆臂效应**：

IMU 固连在相机上、偏离光心 ``r``，则两者的加速度关系是

    a_WI(t) = a_WC(t) + R̈_WC(t) · r           （r 为常向量）

而加速度计测的是比力：``a_meas = R_IW (a_WI − g_W)``。代入消掉 a_WI：

    **R̈_WC(t) · r  =  R_WI(t) · a_meas(t) + g_W − a_WC(t)**

右边全部已知（姿态 + 加速度计 + 由 PnP 位姿二次微分得到的相机加速度），
而 ``r`` 是 3 个未知数、线性出现。对多个时刻堆叠即得最小二乘解。

**这个方法的精度完全取决于"能不能把相机轨迹二阶导数算干净"**，所以：
1. PnP 位姿先做 Savitzky-Golay 平滑再求导；
2. 报告里必须给出 LS 残差，残差大就说明平移不可信，**要老实说**，
   并退回到"用尺子量 + 说明不确定度"。
"""
from __future__ import annotations

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════
#  基础工具
# ══════════════════════════════════════════════════════════════
def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]], float)


def T_from_Rt(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    return T_from_Rt(R.T, -R.T @ t)


def rot_angle(R):
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def R_IC_to_lever(R_IC, t_IC):
    """t_IC -> 杠杆臂 r（IMU 原点在相机系下的坐标）。"""
    return -R_IC.T @ np.asarray(t_IC).reshape(3)


def lever_to_t_IC(R_IC, r):
    """杠杆臂 r -> t_IC。"""
    return -np.asarray(R_IC) @ np.asarray(r).reshape(3)


def T_IC_to_dict(T_IC: np.ndarray) -> dict:
    R, t = T_IC[:3, :3], T_IC[:3, 3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll, pitch, yaw = (np.arctan2(R[2, 1], R[2, 2]),
                            np.arctan2(-R[2, 0], sy),
                            np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll, pitch, yaw = np.arctan2(-R[1, 2], R[1, 1]), np.pi / 2, 0.0
    return {
        "convention": "T_IC_means_camera_to_imu",
        "convention_note": "p_I = R_IC @ p_C + t_IC；T_IC 把相机系下的点变到 IMU 系下",
        "T_IC": [[float(v) for v in row] for row in T_IC],
        "rotation_euler_deg_zyx": [float(v) for v in np.rad2deg([roll, pitch, yaw])],
        "translation_m": [float(v) for v in t],
        "translation_norm_m": float(np.linalg.norm(t)),
        "lever_arm_in_camera_m": [float(v) for v in R_IC_to_lever(R, t)],
    }


# ══════════════════════════════════════════════════════════════
#  ① 旋转
# ══════════════════════════════════════════════════════════════
HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def solve_rotation(R_WI_list, R_CB_list, methods=None,
                   verbose: bool = False) -> dict:
    """用 ``cv2.calibrateHandEye`` 解 R_IC。

    参数
    ----
    R_WI_list : (N,3,3) 每个相机时刻的 ``R_WI``（IMU 姿态，陀螺积分或板子直接给）
    R_CB_list : (N,3,3) 每个相机时刻的 ``R_CB``（PnP 得到）

    返回 dict：R_IC（多方法中位数）、各方法结果、方法间离散度
    （离散度本身就是"多次独立标定对比"的一种）。
    """
    methods = methods or HAND_EYE_METHODS
    R_g2b = [np.asarray(R, float) for R in R_WI_list]        # T_I^W 的旋转
    t_g2b = [np.zeros((3, 1)) for _ in R_g2b]                # 位置拿不到，给 0
    R_t2c = [np.asarray(R, float) for R in R_CB_list]
    t_t2c = [np.zeros((3, 1)) for _ in R_t2c]

    results, errs = {}, {}
    for name, m in methods.items():
        try:
            Rx, _ = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=m)
            Rx = np.asarray(Rx, float)
            # 严格校验：必须是合法旋转矩阵。某些方法在平移输入退化时
            # （我们传的 t_g2b 全是 0）会返回非正交阵甚至 NaN，
            # 直接丢进平均会**把结果整体拖偏几十度**——这个坑踩过一次。
            if not np.isfinite(Rx).all():
                raise ValueError("解里含 NaN/Inf")
            if abs(np.linalg.det(Rx) - 1.0) > 1e-4:
                raise ValueError(f"det={np.linalg.det(Rx):.6f} ≠ 1，不是旋转矩阵")
            if np.abs(Rx @ Rx.T - np.eye(3)).max() > 1e-4:
                raise ValueError("解不是正交阵")
            results[name] = Rx
        except Exception as e:  # noqa: BLE001
            errs[name] = f"{type(e).__name__}: {e}"

    if not results:
        raise RuntimeError(f"所有手眼标定方法都失败：{errs}")

    # 找最大一致簇（互相在 tol 内的解），只用簇内解取平均。
    # 这样个别方法发散也不会污染结果。
    names = list(results)
    tol = 1.0
    best, best_cnt = names[0], -1
    for a in names:
        cnt = sum(1 for b in names
                  if np.rad2deg(rot_angle(results[a].T @ results[b])) < tol)
        if cnt > best_cnt:
            best, best_cnt = a, cnt
    cluster = [results[b] for b in names
               if np.rad2deg(rot_angle(results[best].T @ results[b])) < tol]
    spread = {b: float(np.rad2deg(rot_angle(results[best].T @ results[b])))
              for b in names}

    # 取簇内解的旋转向量平均（不要直接平均矩阵）
    rvs = [cv2.Rodrigues(R)[0].reshape(3) for R in cluster]
    R_avg, _ = cv2.Rodrigues(np.mean(rvs, axis=0).reshape(3, 1))

    if verbose:
        in_cluster = {n for n in names if spread[n] < tol}
        print(f"  有效方法 {len(results)}/{len(methods)}，一致簇 {len(cluster)} 个")
        for n in names:
            print(f"    {n:11s} 相对基准差 {spread[n]:8.4f}°"
                  f"{'  ✅在簇内' if n in in_cluster else '  ❌离群被剔除'}")
        for n, e in errs.items():
            print(f"    {n:11s} 无效解：{e}")
        inner0 = [spread[n] for n in in_cluster]
        print(f"  一致簇内最大离散 {max(inner0):.4f}° "
              f"({'✅ 一致' if max(inner0) < 1.0 else '⚠️'})")

    inner = [spread[n] for n in names if spread[n] < tol]
    return {"R_IC": R_avg, "per_method": results, "failed": errs,
            "cluster": [n for n in names if spread[n] < tol],
            "spread_deg": spread,
            "max_spread_deg": float(max(inner))}


def ax_xb_residual(R_WI_list, R_CB_list, R_IC) -> np.ndarray:
    """逐对验证 ``A X = X B`` 的旋转残差（度）。用于报告里的"运动一致性"。"""
    n = len(R_WI_list)
    out = []
    for i in range(n - 1):
        A = np.asarray(R_WI_list[i + 1]).T @ np.asarray(R_WI_list[i])
        B = np.asarray(R_CB_list[i + 1]) @ np.asarray(R_CB_list[i]).T
        out.append(np.rad2deg(rot_angle(A @ R_IC @ (R_IC @ B).T)))
    return np.asarray(out)


# ══════════════════════════════════════════════════════════════
#  ② 平移（杠杆臂）
# ══════════════════════════════════════════════════════════════
def _savgol(y, dt, window, polyorder, deriv=0, axis=0):
    from scipy.signal import savgol_filter

    w = int(window) | 1                      # 必须是奇数
    w = min(w, (len(y) - 1) | 1)
    w = max(w, polyorder + 2 | 1)
    return savgol_filter(y, w, polyorder, deriv=deriv, delta=dt, axis=axis)


def solve_lever_arm(cam_times, R_CB_list, t_CB_list, imu_times, accel,
                    R_IC, tau: float = 0.0, gravity: float = 9.80665,
                    smooth_window: int = 9, polyorder: int = 3,
                    verbose: bool = False) -> dict:
    """用杠杆臂效应解 IMU 原点在相机系下的坐标 ``r``。

    参数
    ----
    cam_times  : (N,) 相机时间戳（主机时钟）
    R_CB_list  : (N,3,3) PnP 得到的 R_CB
    t_CB_list  : (N,3)   PnP 得到的 t_CB
    imu_times  : (M,) IMU 时间戳（IMU 时钟）
    accel      : (M,3) 加速度计（比力，m/s^2）
    tau        : 时钟偏移，``t_host = t_imu + tau``
    R_IC       : 已解出的旋转外参

    返回 dict：r、t_IC、LS 残差、条件数。
    """
    t = np.asarray(cam_times, float)
    R_CB = np.asarray(R_CB_list, float)
    t_CB = np.asarray(t_CB_list, float)
    n = len(t)
    if n < 12:
        raise ValueError(f"相机帧只有 {n} 帧，二次微分不可靠，至少需要 12 帧")

    dt = float(np.mean(np.diff(t)))
    if np.std(np.diff(t)) > 0.05 * dt:
        if verbose:
            print(f"  ⚠️ 相机时间戳间隔不均匀（std/dt={np.std(np.diff(t))/dt:.3f}），"
                  f"求导前请先按时间重采样")

    # 相机在世界系下的位姿：板=世界 → T_WC = (T_CB)^{-1}
    R_WC = np.array([R.T for R in R_CB])
    p_WC = np.array([-R.T @ tt for R, tt in zip(R_CB, t_CB)])

    # ① 位置：平滑后求二阶导
    p_s = _savgol(p_WC, dt, smooth_window, polyorder, deriv=0)
    a_WC = _savgol(p_s, dt, smooth_window, polyorder, deriv=2)

    # ② 姿态：先由中心差分求角速度，再平滑求角加速度
    w_W = np.zeros((n, 3))
    for i in range(1, n - 1):
        dR = R_WC[i + 1] @ R_WC[i - 1].T
        w_W[i] = cv2.Rodrigues(dR)[0].reshape(3) / (2 * dt)
    w_W[0], w_W[-1] = w_W[1], w_W[-2]
    w_W_s = _savgol(w_W, dt, smooth_window, polyorder, deriv=0)
    alpha_W = _savgol(w_W_s, dt, smooth_window, polyorder, deriv=1)

    # ③ R̈ = R([ω_b]×² + [α_b]×)，ω_b/α_b 为本体系分量
    Rdd = np.zeros((n, 3, 3))
    for i in range(n):
        wb = R_WC[i].T @ w_W_s[i]
        ab = R_WC[i].T @ alpha_W[i]
        Rdd[i] = R_WC[i] @ (skew(wb) @ skew(wb) + skew(ab))

    # ④ 加速度计按主机时钟重采样：t_host = t_imu + tau
    t_imu = np.asarray(imu_times, float) + tau
    a_meas = np.stack([np.interp(t, t_imu, np.asarray(accel, float)[:, j])
                       for j in range(3)], axis=1)

    # ⑤ 组装线性方程 M r = b
    R_WI = np.array([R_WC[i] @ R_IC.T for i in range(n)])
    g_W = np.array([0.0, 0.0, -gravity])
    M = Rdd.reshape(-1, 3)
    b = (np.einsum("nij,nj->ni", R_WI, a_meas) + g_W[None, :] - a_WC).reshape(-1)

    r, *_ = np.linalg.lstsq(M, b, rcond=None)
    resid = M @ r - b
    cond = float(np.linalg.cond(M))

    t_IC = lever_to_t_IC(R_IC, r)
    out = {
        "r_lever_m": r,
        "t_IC_m": t_IC,
        "residual_rms_ms2": float(np.sqrt(np.mean(resid ** 2))),
        "cond": cond,
        "n_samples": int(len(t)),
        "reliable": bool(np.sqrt(np.mean(resid ** 2)) < 1.0 and cond < 1e4),
    }
    if verbose:
        print(f"  杠杆臂 r = {np.array2string(r, precision=4)} m "
              f"(|r|={np.linalg.norm(r)*100:.2f} cm)")
        print(f"  t_IC   = {np.array2string(t_IC, precision=4)} m "
              f"(|t|={np.linalg.norm(t_IC)*100:.2f} cm)")
        print(f"  LS 残差 RMS = {out['residual_rms_ms2']:.4f} m/s²，"
              f"cond(M) = {cond:.1f}  "
              f"{'✅ 可信' if out['reliable'] else '⚠️ 平移不可信，请结合尺子测量并说明'}")
    return out


# ══════════════════════════════════════════════════════════════
#  ③ 运动一致性验证（任务书第 4 节要的指标之一）
# ══════════════════════════════════════════════════════════════
def motion_consistency(R_WI_list, R_CB_list, R_IC) -> dict:
    """相机与 IMU 对同一段运动的描述有多一致 —— 外参对不对的直接证据。

    **必须用相对旋转**，不能用绝对姿态比：

        由 AX = XB，相邻两帧满足  R_WI(2)^T R_WI(1) = R_IC · [R_CB(2) R_CB(1)^T] · R_IC^T

    这个式子两边都是"相对旋转"，**与坐标系原点、朝向都无关**，所以
    陀螺积分缺的 yaw 初值不会影响它。

    ⚠️ 一开始我写成"比较 R_WI 与 R_CB^T R_IC^T"（绝对姿态），那是错的：
    陀螺积分没有绝对 yaw 基准，这个指标会凭空多出几十度的误差。
    这类错误不改定义是修不好的。

    合格参考：均值 < 1~2°。
    """
    e = ax_xb_residual(R_WI_list, R_CB_list, R_IC)
    if len(e) == 0:
        return {"n": 0}
    return {"n": int(len(e)), "mean_deg": float(e.mean()),
            "rms_deg": float(np.sqrt((e ** 2).mean())),
            "max_deg": float(e.max()), "per_frame_deg": [float(v) for v in e]}


def main() -> int:
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from intrinsics import Intrinsics
    from io_data import load_session
    from sync import estimate_bias, estimate_offset, integrate_gyro, resample_to, \
        static_segment
    from target_detect import BoardSpec, detect

    ap = argparse.ArgumentParser(description="C3 Camera-IMU 外参标定")
    ap.add_argument("--session", required=True)
    ap.add_argument("--intrinsics", required=True, help="C1 输出的内参 yaml")
    ap.add_argument("--pattern", default="9x6")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import yaml
    c, r = (int(v) for v in args.pattern.lower().split("x"))
    spec = BoardSpec("chessboard", (c, r), args.square_mm / 1000.0)
    intr = Intrinsics.from_dict(yaml.safe_load(Path(args.intrinsics).read_text()))
    sess = load_session(args.session)
    if sess.imu is None:
        print("❌ 没有 IMU 数据")
        return 2

    # 零偏
    bias = estimate_bias(sess.imu.accel, sess.imu.gyro)
    print(f"零偏标定：{bias}")
    if bias.get("ok"):
        gyro = sess.imu.gyro - bias["gyro_bias"]
        accel = sess.imu.accel - bias["accel_bias"]
    else:
        gyro, accel = sess.imu.gyro, sess.imu.accel

    # PnP
    R_CB, t_CB, ts = [], [], []
    for p, tt in zip(sess.image_paths, sess.image_ts):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        d = detect(gray, spec)
        if not d.found:
            continue
        ok, rvec, tvec = cv2.solvePnP(d.object_points, d.image_points,
                                      intr.K, intr.dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        R_CB.append(cv2.Rodrigues(rvec)[0])
        t_CB.append(tvec.reshape(3))
        ts.append(tt)
    print(f"PnP 成功 {len(R_CB)}/{len(sess.image_paths)} 帧")
    ts = np.asarray(ts)

    # τ
    sync = estimate_offset(ts, R_CB, sess.imu.t, gyro, verbose=True)
    tau = sync["tau"]

    # 陀螺积分得 R_WI，再重采样到相机时刻
    R_WI_all = integrate_gyro(sess.imu.t, gyro)
    R_WI = np.array([R_WI_all[int(np.argmin(np.abs(sess.imu.t - (t - tau))))]
                     for t in ts])

    print("\n--- 旋转外参 ---")
    rot_res = solve_rotation(R_WI, R_CB, verbose=True)
    R_IC = rot_res["R_IC"]

    print("\n--- 运动一致性 ---")
    mc = motion_consistency(R_WI, R_CB, R_IC)
    print(f"  相机 vs IMU 姿态差角：均值 {mc['mean_deg']:.3f}°，"
          f"max {mc['max_deg']:.3f}°")

    print("\n--- 平移外参（杠杆臂）---")
    lev = solve_lever_arm(ts, R_CB, t_CB, sess.imu.t, accel, R_IC,
                          tau=tau, verbose=True)

    T_IC = T_from_Rt(R_IC, lev["t_IC_m"])
    d = T_IC_to_dict(T_IC)
    print(f"\n{'-'*60}\n输出 T_IC（相机系 → IMU 系）:")
    for k, v in d.items():
        print(f"  {k:28s} {v}")

    if args.out:
        from io_data import build_result, save_result
        res = build_result(
            intr.to_dict(), d,
            {"motion_consistency": mc,
             "handeye_spread_deg": rot_res["spread_deg"],
             "ax_xb_residual_deg_max": float(ax_xb_residual(R_WI, R_CB, R_IC).max()),
             "lever_arm": {k: (v.tolist() if hasattr(v, "tolist") else v)
                           for k, v in lev.items()},
             "time_offset": {"tau_s": tau, **{k: v for k, v in sync.items()
                                              if k not in ("grid_taus", "grid_costs")}}})
        for p in save_result(res, args.out):
            print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
