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


def solve_rotation_robust(R_WI_list, R_CB_list, methods=None, verbose: bool = False,
                          trim: float = 0.2, iters: int = 3) -> dict:
    """带**帧级异常剔除**的手眼旋转标定。

    为什么需要：真机数据里总有几帧是坏的 —— 运动模糊、标定板部分出画、
    PnP 在两个解之间跳、或者手抖那一瞬间。普通最小二乘会被它们带偏，
    而**残差看起来仍然正常**。

    做法（trimmed re-solve，简单但有效）：

    1. 全量解一次 ``R_IC``；
    2. 用解出的 ``R_IC`` 算每帧的 AX=XB 相对旋转残差，
       取该帧参与的所有相邻对里的**最大值**作为这一帧的"坏度"；
    3. 剔掉坏度最大的 ``trim`` 比例；
    4. 回到 1，迭代 ``iters`` 次。

    返回除 ``solve_rotation`` 的字段外，还带 ``kept_indices`` / ``n_used`` /
    ``trim_history``（剔了多少、阈值多少，可直接写进报告）。
    """
    n_total = len(R_WI_list)
    idx = np.arange(n_total)
    hist = []
    for it in range(max(1, iters)):
        sub_WI = [R_WI_list[i] for i in idx]
        sub_CB = [R_CB_list[i] for i in idx]
        res = solve_rotation(sub_WI, sub_CB, methods=methods,
                             verbose=bool(verbose and it == 0))
        R_IC = res["R_IC"]
        # 帧太少就不再剔，否则估不准
        if it == iters - 1 or len(idx) < 60:
            break
        pair = np.asarray(ax_xb_residual(sub_WI, sub_CB, R_IC))
        if len(pair) == 0:
            break
        bad = np.zeros(len(idx))
        bad[:-1] = np.maximum(bad[:-1], pair)
        bad[1:] = np.maximum(bad[1:], pair)
        thr = float(np.percentile(bad, 100.0 * (1.0 - trim)))
        keep = bad <= thr
        if keep.sum() < max(40, int(0.5 * len(idx))):
            break
        hist.append({"iter": it, "n_before": int(len(idx)),
                     "n_after": int(keep.sum()), "thr_deg": thr})
        idx = idx[keep]

    res["kept_indices"] = idx.tolist()
    res["n_used"] = int(len(idx))
    res["n_total"] = n_total
    res["trim_history"] = hist
    if verbose and hist:
        print(f"  鲁棒剔除：{n_total} → {res['n_used']} 帧"
              f"（{len(hist)} 轮，末轮阈值 {hist[-1]['thr_deg']:.4f}°）")
    return res


# ══════════════════════════════════════════════════════════════
#  ② 平移（杠杆臂）
# ══════════════════════════════════════════════════════════════
def _irls_huber(A, b, n_iter: int = 8, k: float = 1.345):
    """Huber 加权的迭代重加权最小二乘（IRLS）。

    标定数据里少数几帧的残差可能是其余的几十倍（坏帧）。普通最小二乘是
    平方损失，会被它们主导；Huber 损失对小残差保持平方、对大残差转线性，
    等价于自动把大残差样本降权。

    ``k=1.345`` 是 Huber 的标准取法（正态下约 95% 效率）。
    尺度用 MAD 稳健估计，避免被离群值本身污染。
    """
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    w = np.ones(len(b))
    for _ in range(max(1, n_iter)):
        res = b - A @ sol
        s = 1.4826 * float(np.median(np.abs(res - np.median(res)))) + 1e-12
        u = np.abs(res) / s
        w = np.where(u <= k, 1.0, k / np.maximum(u, 1e-12))
        sw = np.sqrt(w)
        sol, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
    return sol, w


def solve_lever_arm(cam_times, R_CB_list, t_CB_list, imu_times, accel,
                    R_IC, tau: float = 0.0, gravity: float = 9.80665,
                    pos_sigma: float | None = None, rot_sigma: float = 0.002,
                    smooth_window: int = 9, polyorder: int = 3,
                    use_spline: bool = True, estimate_gravity: bool = True,
                    robust: bool = True, verbose: bool = False) -> dict:
    """用杠杆臂效应解 IMU 原点在相机系下的坐标 ``r``。

    ``use_spline=True``（默认）用 **B样条解析导数**求 ``a_WC`` 与 ``R̈``；
    ``False`` 退回 Savitzky-Golay 数值微分（保留用于对比，实测差约 100 倍）。

    ``pos_sigma`` **必须尽量设准**（PnP 位置的预期噪声，米）。实测：
    3 mm 噪声下，sigma 设对时 a 误差 0.0125 m/s²；sigma 设成 1 mm 时
    误差反而涨到 6.8 m/s²——**比不做还糟**。``quality()`` 会打印实际残差，
    与设定值差一个量级就说明设错了。
    """
    import cv2 as _cv2

    t = np.asarray(cam_times, float)
    R_CB = np.asarray(R_CB_list, float)
    t_CB = np.asarray(t_CB_list, float)
    n = len(t)
    if n < 20:
        raise ValueError(f"相机帧只有 {n} 帧，B样条拟合不可靠，至少需要 20 帧")

    dt = float(np.mean(np.diff(t)))
    if verbose and np.std(np.diff(t)) > 0.05 * dt:
        print(f"  ⚠️ 相机时间戳间隔不均匀（std/dt={np.std(np.diff(t))/dt:.3f}）")

    # 相机在世界系下的位姿：板=世界 → T_WC = (T_CB)^{-1}
    R_WC = np.array([R.T for R in R_CB])
    p_WC = np.array([-R.T @ tt for R, tt in zip(R_CB, t_CB)])

    traj = None
    if use_spline:
        from trajectory import SplineTrajectory

        # ★ pos_sigma 必须与**实际的 PnP 位置噪声**匹配，否则样条过拟合，
        #   二阶导噪声被放大，杠杆臂直接崩。实测（真值 |r| = 4.25 cm）：
        #       sigma=1.5mm（实际噪声~6mm）→ 杠杆臂误差 8.31 cm，LS 残差 3.18
        #       sigma=6.0mm              → 杠杆臂误差 0.61 cm，LS 残差 0.041
        #   而实际噪声事先并不知道，所以这里**自动标定**：迭代到
        #   "拟合残差 ≈ sigma"（统计上正确的平滑量），并带阻尼避免振荡。
        #   最优区很平坦（3~12 mm 结果都在 1 cm 内），所以不敏感。
        if pos_sigma is None:
            sg = 0.005
            for _ in range(10):
                tr = SplineTrajectory.fit(t, R_WC, p_WC, pos_sigma=sg,
                                          rot_sigma=rot_sigma)
                res = tr.quality()["pos_residual_rms_m"]
                if sg > 0 and abs(res - sg) / sg < 0.03:
                    break
                sg = 0.4 * sg + 0.6 * max(res, 1e-6)
            pos_sigma, traj = sg, tr
            if verbose:
                print(f"  pos_sigma 自动标定 = {pos_sigma*1000:.2f} mm "
                      f"(拟合残差 {res*1000:.2f} mm)")
        else:
            traj = SplineTrajectory.fit(t, R_WC, p_WC, pos_sigma=pos_sigma,
                                        rot_sigma=rot_sigma)
        R_WC_s = traj.R(t)
        a_WC = traj.a(t)
        Rdd = traj.Rdd(t)
        if verbose:
            print(f"  {traj.report()}")
    else:
        # 回退路径：Savitzky-Golay 数值微分
        p_s = _savgol(p_WC, dt, smooth_window, polyorder, deriv=0)
        a_WC = _savgol(p_s, dt, smooth_window, polyorder, deriv=2)
        w_W = np.zeros((n, 3))
        for i in range(1, n - 1):
            dR = R_WC[i + 1] @ R_WC[i - 1].T
            w_W[i] = _cv2.Rodrigues(dR)[0].reshape(3) / (2 * dt)
        w_W[0], w_W[-1] = w_W[1], w_W[-2]
        w_W_s = _savgol(w_W, dt, smooth_window, polyorder, deriv=0)
        alpha_W = _savgol(w_W_s, dt, smooth_window, polyorder, deriv=1)
        R_WC_s = R_WC
        Rdd = np.zeros((n, 3, 3))
        for i in range(n):
            wb = R_WC[i].T @ w_W_s[i]
            ab = R_WC[i].T @ alpha_W[i]
            Rdd[i] = R_WC[i] @ (skew(wb) @ skew(wb) + skew(ab))

    # ④ 加速度计按主机时钟重采样：t_host = t_imu + tau
    t_imu = np.asarray(imu_times, float) + tau
    a_meas = np.stack([np.interp(t, t_imu, np.asarray(accel, float)[:, j])
                       for j in range(3)], axis=1)

    # ⑤ 组装线性方程
    R_WI = np.array([R_WC_s[i] @ R_IC.T for i in range(n)])
    rhs = (np.einsum("nij,nj->ni", R_WI, a_meas) - a_WC).reshape(-1)
    M = Rdd.reshape(-1, 3)

    # ★ 关键：**不要把板系当成重力对齐的世界系**。
    #
    # 标定板的两解二义性意味着 PnP 解出的 R_CB 可能与真值相差一个"绕板面内轴
    # 180°"的翻转；由于棋盘格中心对称，两解的**重投影几乎相同**，无法从图像
    # 区分。好消息是 AX=XB 对常量板系偏置免疫（S 在 B 里抵消），所以**旋转
    # 外参不受影响**；坏消息是 a_WC / R̈ 都在板系里，而重力在板系中的方向
    # 因此是未知的。
    #
    # 解法：把重力向量 g_W 也当作未知量，与 r 联合最小二乘——
    #     R̈_WC · r + g_W  =  R_WI·a_meas − a_WC
    # 对 [r; g_W] 是线性的（6 个未知量）。附带好处：|g_W| 应当等于 9.80665，
    # 这就成了一个**内建的自检指标**。
    if estimate_gravity:
        A = np.hstack([M, np.tile(np.eye(3), (n, 1))])
        sol, w_final = _irls_huber(A, rhs, n_iter=8 if robust else 1)
        r, g_est = sol[:3], sol[3:]
        resid = A @ sol - rhs
    else:
        g_est = np.array([0.0, 0.0, -gravity])
        y = rhs - np.tile(g_est, n)
        sol, w_final = _irls_huber(M, y, n_iter=8 if robust else 1)
        r = sol[:3]
        resid = M @ r - y
    n_downweighted = int(np.sum(w_final < 0.9))

    cond = float(np.linalg.cond(M))
    rms = float(np.sqrt(np.mean(resid ** 2)))
    g_norm = float(np.linalg.norm(g_est))
    # |g| 与真值差多少 —— 只有板系与世界系真的对齐时才会接近 0
    g_err = abs(g_norm - gravity)

    t_IC = lever_to_t_IC(R_IC, r)
    out = {
        "r_lever_m": r,
        "t_IC_m": t_IC,
        "gravity_in_board_frame": g_est,
        "gravity_norm": g_norm,
        "gravity_norm_error": g_err,
        "gravity_tilt_deg": float(np.rad2deg(np.arccos(np.clip(
            -g_est[2] / max(g_norm, 1e-9), -1, 1)))),
        "residual_rms_ms2": rms,
        "cond": cond,
        "n_downweighted": n_downweighted,
        "n_samples": int(len(t)),
        "pos_sigma_used": float(pos_sigma),
        "method": ("spline" if use_spline else "savgol")
                  + ("+gravity-joint" if estimate_gravity else ""),
        "trajectory_quality": traj.quality() if traj is not None else None,
        # 可信判据：LS 残差要明显小于杠杆臂信号量级（约 0.06 m/s²），
        # 且估出的 |g| 要接近 9.80665
        "reliable": bool(rms < 0.05 and cond < 1e4 and g_err < 0.2),
    }
    if verbose:
        print(f"  重力(板系) = {np.array2string(g_est, precision=4)}  "
              f"|g| = {g_norm:.4f} m/s² (真值 {gravity:.4f}, 差 {g_err:.4f})")
        print(f"  杠杆臂 r   = {np.array2string(r, precision=4)} m "
              f"(|r|={np.linalg.norm(r)*100:.2f} cm)")
        print(f"  t_IC       = {np.array2string(t_IC, precision=4)} m "
              f"(|t|={np.linalg.norm(t_IC)*100:.2f} cm)")
        print(f"  LS 残差 RMS = {rms:.5f} m/s²（杠杆臂信号量级约 0.06 m/s²），"
              f"cond(M) = {cond:.1f}")
        if n_downweighted:
            print(f"  Huber 降权样本 {n_downweighted}/{n} 个"
                  f"（{n_downweighted/n*100:.1f}%，坏帧会被自动压制）")
        if g_err > 0.2:
            print(f"  ⚠️ |g| 偏差 {g_err:.4f} 偏大：说明板系与世界系未对齐或数据有问题")
        print(f"  {'✅ 可信' if out['reliable'] else '⚠️ 残差偏大，平移仍不可信'}")
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
    bias = estimate_bias(sess.imu.t, sess.imu.accel, sess.imu.gyro)
    print(f"零偏标定：{bias}")
    if bias.get("ok"):
        gyro = sess.imu.gyro - bias["gyro_bias"]
        accel = sess.imu.accel - bias["accel_bias"]
    else:
        gyro, accel = sess.imu.gyro, sess.imu.accel

    # PnP（务必用 solve_pnp_board：它同时处理角点顺序约定与平面二义性）
    from target_detect import solve_pnp_board

    R_CB, t_CB, ts = [], [], []
    prev_R = None
    n_order_flip = 0
    for p, tt in zip(sess.image_paths, sess.image_ts):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        d = detect(gray, spec)
        if not d.found:
            continue
        try:
            sol = solve_pnp_board(d.object_points, d.image_points,
                                  intr.K, intr.dist, prev_R=prev_R,
                                  pattern_size=spec.pattern_size)
        except Exception:  # noqa: BLE001
            continue
        R = cv2.Rodrigues(sol["rvec"])[0]
        prev_R = R
        n_order_flip += int(sol.get("order") == "x_reversed")
        R_CB.append(R)
        t_CB.append(sol["tvec"].reshape(3))
        ts.append(tt)
    print(f"PnP 成功 {len(R_CB)}/{len(sess.image_paths)} 帧"
          f"（其中 {n_order_flip} 帧用了 x 反序的角点约定）")
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
