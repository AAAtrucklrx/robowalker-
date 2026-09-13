#!/usr/bin/env python3
"""B样条轨迹：把离散的 PnP 位姿变成**可解析求导的连续轨迹**。

为什么需要这个模块
------------------------------------------------------------------
C3 的平移外参（杠杆臂）方程是：

    R̈_WC(t) · r  =  R_WI(t)·a_meas(t) + g_W − a_WC(t)

右边要 `a_WC`（相机加速度）。之前我用 Savitzky-Golay 对 PnP 位置做**数值二阶微分**，
结果残差 11 m/s²（≈g）——因为：

* 平面标定板的**深度方向可观测性天生就差**，PnP 位置噪声偏大；
* 二阶微分对噪声的放大是 **1/Δt²** 量级，直接淹没了杠杆臂那点信号
  （`|b|` 只有 0.059 m/s²，而 `|R_WI·a_meas|` 是 9.82 —— **160:1 的抵消**）。

**解法**：把位姿序列拟合成 **三次 B样条**，再用样条的**解析导数**。
样条本身就带低通性质（拟合噪声得到光滑曲线），导数又是解析的，
不再有差分噪声放大。

旋转部分的做法
------------------------------------------------------------------
旋转不能直接对旋转矩阵元素拟合（拟合完不是正交阵）。这里用：

1. 把每帧的 R 转成四元数；
2. **半球对齐**（相邻四元数点积为负就取反）——否则 q 与 −q 表示同一旋转这件事
   会让序列出现人为跳变，拟合直接崩；
3. 对 4 个分量分别拟合 B样条得到 q̃(t)（**未归一化**）；
4. 按 s = |q̃| 归一化，并用链式法则写出解析的 q̇、q̈：

       q̇  = u̇/s − u(u·u̇)/s³
       q̈  = ü/s − 2u̇(u·u̇)/s³ − u(u̇·u̇ + u·ü)/s³ + 3u(u·u̇)²/s⁵

5. 由 q̇、q̈ 得到角速度/角加速度：

       ω_world = 2·Im(q̇ ⊗ q*)
       α_world = 2·Im(q̈ ⊗ q* + q̇ ⊗ q̇*)

6. 再用 `R̈ = R([ω_b]×² + [α_b]×)` 构造旋转矩阵的二阶导。

用法::

    traj = SplineTrajectory(t, R_WC_list, p_WC_list, pos_sigma=0.0015)
    R  = traj.R(tq);  p = traj.p(tq)
    Rdd = traj.Rdd(tq);  a = traj.a(tq)
    info = traj.quality()      # 拟合残差，判断 sigma 设得合不合适
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import UnivariateSpline


# ══════════════════════════════════════════════════════════════
#  四元数工具
# ══════════════════════════════════════════════════════════════
def quat_from_R(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 四元数 (w, x, y, z)。"""
    m = np.asarray(R, float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s,
                      (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s,
                      (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                      (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                      0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_mul(a, b):
    wa, xa, ya, za = a
    wb, xb, yb, zb = b
    return np.array([
        wa * wb - xa * xb - ya * yb - za * zb,
        wa * xb + xa * wb + ya * zb - za * yb,
        wa * yb - xa * zb + ya * wb + za * xb,
        wa * zb + xa * yb - ya * xb + za * wb,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def hemisphere_align(qs: np.ndarray) -> np.ndarray:
    """四元数半球对齐：相邻点积为负就整体取反。

    必须做这一步。q 与 −q 表示同一个旋转，但四元数序列里一旦出现符号翻转，
    逐分量拟合就会在那一帧产生巨大跳变，样条会被带飞。
    """
    out = np.array(qs, float).copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0:
            out[i] = -out[i]
    return out


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], float)


def _fit_spline(t, y, sigma, k=3):
    """按噪声水平 sigma 做平滑样条拟合。

    ``UnivariateSpline`` 的平滑参数 s 是"残差平方和的容忍上限"，
    所以 s = n·σ² 正好对应"允许每个样本偏离 σ"。
    sigma <= 0 时退化为插值（s=0）。
    """
    n = len(t)
    s = 0.0 if sigma is None or sigma <= 0 else n * float(sigma) ** 2
    return UnivariateSpline(t, np.asarray(y, float), k=k, s=s)


# ══════════════════════════════════════════════════════════════
@dataclass
class SplineTrajectory:
    """相机轨迹（板=世界系）：``R_WC`` 与 ``p_WC`` 的 C² 连续表示。"""

    t: np.ndarray
    splines_pos: list
    splines_quat: list
    pos_sigma: float
    rot_sigma: float
    p_data: np.ndarray = None      # 原始（含噪）位置，用于算拟合残差
    q_data: np.ndarray = None      # 原始四元数

    # -- 构造 --------------------------------------------------
    @classmethod
    def fit(cls, t, R_WC_list, p_WC_list, pos_sigma: float = 0.0015,
            rot_sigma: float = 0.002, k: int = 3) -> "SplineTrajectory":
        """拟合轨迹。

        ``pos_sigma``：PnP 位置的预期噪声（米）。平面目标深度噪声大，
        一般 1~5 mm 量级。**这个值设错会直接影响外参结果**，所以
        ``quality()`` 会把实际残差打出来供核对。

        ``rot_sigma``：姿态噪声（弧度），一般 0.001~0.005 rad。
        """
        t = np.asarray(t, float)
        if len(t) < 2 * k + 2:
            raise ValueError(f"样本数 {len(t)} 太少，三次样条至少要 {2*k+2} 个")

        p = np.asarray(p_WC_list, float)
        sp = [_fit_spline(t, p[:, j], pos_sigma, k) for j in range(3)]

        qs = hemisphere_align(np.array([quat_from_R(R) for R in R_WC_list]))
        sq = [_fit_spline(t, qs[:, j], rot_sigma, k) for j in range(4)]

        return cls(t=t, splines_pos=sp, splines_quat=sq,
                   pos_sigma=pos_sigma, rot_sigma=rot_sigma,
                   p_data=p, q_data=qs)

    # -- 求值 --------------------------------------------------
    def p(self, tq, deriv: int = 0) -> np.ndarray:
        tq = np.atleast_1d(np.asarray(tq, float))
        cols = [s(tq, nu=deriv) if deriv else s(tq) for s in self.splines_pos]
        return np.stack(cols, axis=1)

    def a(self, tq) -> np.ndarray:
        """相机加速度 p̈_WC。解析二阶导。"""
        return self.p(tq, deriv=2)

    def quat_raw(self, tq, deriv: int = 0) -> np.ndarray:
        tq = np.atleast_1d(np.asarray(tq, float))
        cols = [s(tq, nu=deriv) if deriv else s(tq) for s in self.splines_quat]
        return np.stack(cols, axis=1)

    def R(self, tq) -> np.ndarray:
        u = self.quat_raw(tq, 0)
        return np.array([quat_to_R(u[i]) for i in range(len(u))])

    def omega_alpha(self, tq):
        """返回 ``(ω_world, α_world)``：世界系角速度与角加速度。"""
        u = self.quat_raw(tq, 0)
        du = self.quat_raw(tq, 1)
        ddu = self.quat_raw(tq, 2)
        n = len(u)
        w_w = np.zeros((n, 3))
        a_w = np.zeros((n, 3))
        for i in range(n):
            ui, dui, ddui = u[i], du[i], ddu[i]
            s = np.linalg.norm(ui)
            if s < 1e-12:
                continue
            s3, s5 = s ** 3, s ** 5
            udu = float(ui @ dui)
            # 归一化后的解析导数（链式法则）
            q = ui / s
            qd = dui / s - ui * udu / s3
            qdd = (ddui / s - 2 * dui * udu / s3
                   - ui * (float(dui @ dui) + float(ui @ ddui)) / s3
                   + 3 * ui * udu ** 2 / s5)
            # ω_world = 2·Im(q̇ ⊗ q*)，α_world = 2·Im(q̈ ⊗ q* + q̇ ⊗ q̇*)
            pq = quat_mul(qd, quat_conj(q))
            pa = quat_mul(qdd, quat_conj(q)) + quat_mul(qd, quat_conj(qd))
            w_w[i] = 2 * pq[1:]
            a_w[i] = 2 * pa[1:]
        return w_w, a_w

    def Rdd(self, tq) -> np.ndarray:
        """旋转矩阵的二阶导 ``R̈ = R([ω_b]×² + [α_b]×)``。"""
        tq = np.atleast_1d(np.asarray(tq, float))
        R = self.R(tq)
        w_w, a_w = self.omega_alpha(tq)
        out = np.zeros((len(tq), 3, 3))
        for i in range(len(tq)):
            wb = R[i].T @ w_w[i]
            ab = R[i].T @ a_w[i]
            out[i] = R[i] @ (skew(wb) @ skew(wb) + skew(ab))
        return out

    # -- 体检 --------------------------------------------------
    def quality(self) -> dict:
        """拟合残差。**用来核对 sigma 设得对不对**：
        实际残差应当与设定的 sigma 同量级；差一个量级就说明 sigma 设错了。
        """
        pf = self.p(self.t)
        pos_res = float(np.sqrt(np.mean(np.sum((pf - self.p_data) ** 2, axis=1))))
        # 姿态残差：拟合四元数（归一化后）与原始四元数的夹角
        uf = self.quat_raw(self.t, 0)
        uf = uf / np.maximum(np.linalg.norm(uf, axis=1, keepdims=True), 1e-12)
        dots = np.clip(np.abs(np.sum(uf * self.q_data, axis=1)), 0, 1)
        rot_res = float(np.mean(2 * np.arccos(dots)))
        return {
            "pos_residual_rms_m": pos_res,
            "rot_residual_rms_rad": rot_res,
            "n_samples": int(len(self.t)),
            "t_span_s": float(self.t[-1] - self.t[0]),
            "pos_sigma": self.pos_sigma,
            "rot_sigma": self.rot_sigma,
        }

    def report(self) -> str:
        q = self.quality()
        return (f"B样条轨迹: {q['n_samples']} 样本 / {q['t_span_s']:.2f} s，"
                f"位置残差 RMS {q['pos_residual_rms_m']*1000:.3f} mm "
                f"(设定 sigma {self.pos_sigma*1000:.3f} mm)，"
                f"姿态残差 {np.rad2deg(q['rot_residual_rms_rad']):.4f}°")


def main() -> int:
    """自检：用解析轨迹生成位姿 → 加噪 → 拟合 → 检查导数精度。"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from simulate import Trajectory

    traj = Trajectory(rot_amp=np.array([0.20, 0.15, 0.26]),
                      rot_freq=np.array([2.6, 3.6, 1.5]),
                      pos_amp=np.array([0.05, 0.04, 0.03]),
                      center=np.array([0.0, 0.0, 1.30]))
    t = np.linspace(0.0, 6.0, 120)
    R = np.array([traj.R_WC(x) for x in t])
    p = np.array([traj.p_WC(x) for x in t])

    print("=== B样条轨迹自检 ===")
    print(f"样本 {len(t)} 个，{t[-1]-t[0]:.1f} s\n")
    print(f"{'噪声':>10} {'sigma设定':>10} {'位置残差':>12} {'a误差':>10} {'Rdd误差':>10}")
    rng = np.random.default_rng(0)
    for noise_mm, sigma_mm in [(0.0, 0.0), (1.0, 1.0), (3.0, 3.0), (3.0, 1.0)]:
        pn = p + rng.normal(0, noise_mm / 1000, p.shape)
        st = SplineTrajectory.fit(t, R, pn, pos_sigma=sigma_mm / 1000,
                                  rot_sigma=0.0)
        # 真值导数
        a_true = np.array([traj.derivatives(x)["a"] for x in t])
        Rdd_true = np.array([traj.derivatives(x)["Rdd"] for x in t])
        a_err = np.linalg.norm(st.a(t) - a_true, axis=1).mean()
        Rdd_err = np.linalg.norm(st.Rdd(t) - Rdd_true, axis=(1, 2)).mean()
        res = st.quality()["pos_residual_rms_m"]
        print(f"{noise_mm:8.1f}mm {sigma_mm:8.1f}mm {res*1000:10.3f}mm "
              f"{a_err:10.4f} {Rdd_err:10.4f}")
    print("\n说明：位置噪声 0 时 a 误差应接近机器精度；")
    print("      加噪后 a 误差应远小于'直接二阶差分'（后者约 noise/dt^2）。")
    dt = t[1] - t[0]
    print(f"      参考：直接二阶差分在 3mm 噪声下的误差量级 ≈ "
          f"{3e-3/dt**2:.2f} m/s²")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
