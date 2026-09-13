#!/usr/bin/env python3
"""合成装置：造一份**已知真值**的 Camera-IMU 数据，用来验证 C2/C3/C4。

为什么值得单独写这个模块
------------------------------------------------------------------
C1（内参）可以只靠合成图像验证，因为输入输出都是几何。C2/C3 更难：
时间偏移 τ、外参 T_IC、IMU 零偏**同时在起作用**，真机上"结果不对"
根本无法定位是哪一个环节。

所以这里先造一个**真值完全已知**的数字孪生：

* 一条解析轨迹（正弦叠加），可以求出任意阶导数 → 相机位姿、速度、
  加速度、角速度、角加速度**都是精确值**，不受数值微分噪声影响；
* 相机 + IMU 用一个已知的 ``T_IC`` 刚性固连，按已知频率采样；
* IMU 时钟相对主机时钟有一个**已知常量偏移 τ**（真机上这是最麻烦的未知量）；
* 可选地加零偏与噪声，逼近平价硬件的真实情况。

有了它，任何 C2/C3 的方法都可以先在"已知答案"上跑一遍：如果连真值都
解不回来，就不必去怀疑真实数据了。

坐标系与符号
------------------------------------------------------------------
* ``W`` 世界系：标定板固定不动，就取板系为世界系（``T_WB = I``）
* ``C`` 相机系，``I`` IMU 系
* ``T_IC``：**相机系 → IMU 系**（与 README 一致），``p_I = R_IC p_C + t_IC``
* 杠杆臂 ``r`` = IMU 原点在**相机系**下的坐标 = ``-R_IC^T t_IC`` = ``T_CI`` 的平移部分
* 时间：``t_host = t_imu + tau``，即 ``tau`` 是"加到 IMU 时间戳上就换成主机时钟"的量
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

GRAVITY = 9.80665


# ══════════════════════════════════════════════════════════════
#  渲染（与 tools/selftest_intrinsics.py 共用同一份实现）
# ══════════════════════════════════════════════════════════════
def scaled_K(K, ss: int) -> np.ndarray:
    """超采样 ss 倍后的内参。原图 (u,v) 对应 ss 倍图的 (ss*u+(ss-1)/2, ...)。"""
    Ks = np.asarray(K, float).copy()
    Ks[0, 0] *= ss
    Ks[1, 1] *= ss
    Ks[0, 2] = K[0, 2] * ss + (ss - 1) / 2.0
    Ks[1, 2] = K[1, 2] * ss + (ss - 1) / 2.0
    return Ks


def render_board_view(board_spec, K, dist, rvec, tvec, image_size,
                      supersample: int = 3, bg: int = 255,
                      noise_sigma: float = 0.0, rng=None):
    """渲染一帧棋盘格：逐黑方块正向投影 + fillPoly，再降采样抗锯齿。

    ``rvec/tvec`` 是 **板系 -> 相机系** 的位姿（即 T_CB）。
    """
    W, H = image_size
    ss = max(1, int(supersample))
    Ks = scaled_K(K, ss)
    img = np.full((H * ss, W * ss), bg, np.uint8)

    ncols = board_spec.pattern_size[0] + 1
    nrows = board_spec.pattern_size[1] + 1
    s = board_spec.square_size
    quads = []
    for r in range(nrows):
        for c in range(ncols):
            if (r + c) % 2:                       # 只画黑格
                continue
            # 板系原点在第一个内角点，故物理板面左上角在 (-s, -s)
            x0, y0 = (c - 1) * s, (r - 1) * s
            quads.append([[x0, y0, 0], [x0 + s, y0, 0],
                          [x0 + s, y0 + s, 0], [x0, y0 + s, 0]])
    proj, _ = cv2.projectPoints(np.asarray(quads, np.float32).reshape(-1, 3),
                                rvec, tvec, Ks, dist)
    for q in proj.reshape(-1, 4, 2):
        cv2.fillPoly(img, [np.round(q).astype(np.int32)], 0)
    if ss > 1:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    if noise_sigma > 0:
        g = rng if rng is not None else np.random
        img = np.clip(img.astype(np.float32)
                      + g.normal(0, noise_sigma, img.shape), 0, 255).astype(np.uint8)
    return img


# ══════════════════════════════════════════════════════════════
#  轨迹：正弦叠加，任意阶导数都精确可得
# ══════════════════════════════════════════════════════════════
@dataclass
class Trajectory:
    """相机在世界系下的平滑运动。

    旋转用**旋转向量**做正弦叠加再指数映射，保证 R(t) 永远是正交阵；
    平移直接正弦叠加。这样第一/第二阶导数都能用中心差分以 1e-9 精度求出，
    比在真实数据上对 PnP 位姿做数值微分干净得多。
    """

    rot_amp: np.ndarray = field(default_factory=lambda: np.array([0.55, 0.40, 0.70]))
    rot_freq: np.ndarray = field(default_factory=lambda: np.array([1.5, 2.1, 0.9]))
    rot_phase: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.7, 1.9]))
    pos_amp: np.ndarray = field(default_factory=lambda: np.array([0.10, 0.08, 0.06]))
    pos_freq: np.ndarray = field(default_factory=lambda: np.array([0.8, 1.2, 0.6]))
    pos_phase: np.ndarray = field(default_factory=lambda: np.array([0.3, 1.1, 2.4]))
    center: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.70]))
    # 相机基准朝向：把相机 z 轴翻到世界 -z，这样相机在 (0,0,+z) 时是
    # **朝着世界原点的标定板看**。少了这一项，相机会背对板子，
    # 投影变成"从背后透视"，PnP 解出的位姿是镜像的 —— 而且不会报错，
    # 只会让 C2/C3 悄悄崩掉。（C1 的 random_pose 显式把板放在正前方，
    # 所以没暴露出这个问题。）
    R_base: np.ndarray = field(default_factory=lambda: np.diag([1.0, -1.0, -1.0]))

    def rotvec(self, t):
        return self.rot_amp * np.sin(self.rot_freq * t + self.rot_phase)

    def R_WC(self, t) -> np.ndarray:
        """相机系 -> 世界系 的旋转。"""
        R, _ = cv2.Rodrigues(self.rotvec(t))
        return R @ self.R_base

    def p_WC(self, t) -> np.ndarray:
        return self.center + self.pos_amp * np.sin(self.pos_freq * t + self.pos_phase)

    # -- 导数（中心差分；轨迹是解析的，所以差分精度极高）------
    def derivatives(self, t, h1: float = 1e-4, h2: float = 2e-3):
        R = self.R_WC(t)
        p = self.p_WC(t)

        # 角速度：log(R(t+h) R(t-h)^T)/(2h) 给出世界系下的角速度
        dR = self.R_WC(t + h1) @ self.R_WC(t - h1).T
        rv, _ = cv2.Rodrigues(dR)
        w_W = rv.reshape(3) / (2 * h1)

        # 角加速度
        w1 = cv2.Rodrigues(self.R_WC(t + h1) @ self.R_WC(t).T)[0].reshape(3) / h1
        w2 = cv2.Rodrigues(self.R_WC(t) @ self.R_WC(t - h1).T)[0].reshape(3) / h1
        alpha_W = (w1 - w2) / h1

        # 速度、加速度、旋转矩阵的二阶导
        v_W = (self.p_WC(t + h1) - self.p_WC(t - h1)) / (2 * h1)
        a_W = (self.p_WC(t + h2) - 2 * p + self.p_WC(t - h2)) / (h2 ** 2)
        Rdd = (self.R_WC(t + h2) - 2 * R + self.R_WC(t - h2)) / (h2 ** 2)

        return {"R": R, "p": p, "v": v_W, "a": a_W,
                "w_W": w_W, "alpha_W": alpha_W, "Rdd": Rdd}


# ══════════════════════════════════════════════════════════════
#  装置
# ══════════════════════════════════════════════════════════════
def rotation_from_euler_deg(roll, pitch, yaw):
    """ZYX 内旋，返回 3x3。方便用"看着像"的角度描述真值外参。"""
    r, p, y = np.deg2rad([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


@dataclass
class SimRig:
    """一台"数字孪生"的相机+IMU 刚性体。"""

    # ── 真值外参：相机系 -> IMU 系 ──────────────────────────
    R_IC: np.ndarray = field(default_factory=lambda: rotation_from_euler_deg(2.5, -1.8, 88.0))
    t_IC: np.ndarray = field(default_factory=lambda: np.array([0.012, -0.035, 0.021]))

    # ── 时钟：t_host = t_imu + tau ──────────────────────────
    tau: float = 0.137

    # ── 相机内参真值 ────────────────────────────────────────
    image_size: tuple = (1280, 960)
    K: np.ndarray = field(default_factory=lambda: np.array(
        [[1100.0, 0, 628.0], [0, 1098.0, 488.0], [0, 0, 1.0]]))
    dist: np.ndarray = field(default_factory=lambda: np.array(
        [-0.12, 0.045, 0.0012, -0.0008, 0.0]))

    # ── IMU 噪声/零偏（默认给一组"平价 MEMS"的典型值）───────
    gyro_noise: float = 0.0015      # rad/s
    accel_noise: float = 0.020      # m/s^2
    gyro_bias: np.ndarray = field(default_factory=lambda: np.array([0.0040, -0.0022, 0.0015]))
    accel_bias: np.ndarray = field(default_factory=lambda: np.array([0.055, -0.031, 0.048]))

    traj: Trajectory = field(default_factory=Trajectory)

    # -- 派生量 ------------------------------------------------
    @property
    def R_CI(self):
        """IMU 系 -> 相机系 的旋转（T_CI = T_IC^{-1}）。"""
        return self.R_IC.T

    @property
    def t_CI(self):
        return -self.R_IC.T @ self.t_IC

    @property
    def r_lever(self):
        """杠杆臂：IMU 原点在相机系下的坐标（= t_CI）。"""
        return self.t_CI

    def T_IC(self):
        T = np.eye(4)
        T[:3, :3] = self.R_IC
        T[:3, 3] = self.t_IC
        return T

    # -- 位姿 --------------------------------------------------
    def board_pose_in_camera(self, t):
        """返回 (rvec, tvec)，表示 **板系 -> 相机系**（T_CB）。

        板系 = 世界系（板固定不动），所以 T_CB = (T_WC)^{-1}。
        """
        R = self.traj.R_WC(t)
        p = self.traj.p_WC(t)
        T_WC = np.eye(4)
        T_WC[:3, :3] = R
        T_WC[:3, 3] = p
        T_CW = np.linalg.inv(T_WC)
        rvec, _ = cv2.Rodrigues(T_CW[:3, :3])
        return rvec.reshape(3), T_CW[:3, 3].copy()

    def T_WC(self, t):
        T = np.eye(4)
        T[:3, :3] = self.traj.R_WC(t)
        T[:3, 3] = self.traj.p_WC(t)
        return T

    # -- IMU ---------------------------------------------------
    def imu_sample(self, t_host, rng=None, with_noise=True):
        """给定**主机时钟**时刻，算出该时刻 IMU 的陀螺/加速度读数。"""
        d = self.traj.derivatives(t_host)
        R_WC, p_WC, a_WC, Rdd = d["R"], d["p"], d["a"], d["Rdd"]
        w_W, alpha_W = d["w_W"], d["alpha_W"]

        # IMU 在相机上的刚性连接：姿态 = R_WC @ R_CI，位置 = p_WC + R_WC @ r
        R_WI = R_WC @ self.R_CI
        a_WI = a_WC + Rdd @ self.r_lever

        gyro = R_WI.T @ w_W                     # 陀螺量的是 IMU 本体系下的角速度
        g_W = np.array([0.0, 0.0, -GRAVITY])
        accel = R_WI.T @ (a_WI - g_W)           # 加速度计量的是比力

        if with_noise:
            g = rng if rng is not None else np.random
            gyro = gyro + self.gyro_bias + g.normal(0, self.gyro_noise, 3)
            accel = accel + self.accel_bias + g.normal(0, self.accel_noise, 3)
        return gyro, accel, {"R_WI": R_WI, "a_WI": a_WI, "w_W": w_W,
                             "alpha_W": alpha_W, "a_WC": a_WC, "R_WC": R_WC,
                             "p_WC": p_WC}

    def make_imu_stream(self, t_start, t_end, rate=1000.0, rng=None,
                        with_noise=True):
        """生成 IMU 序列。

        返回 ``(t_host, t_imu, gyro, accel)``：``t_imu`` 是 IMU 自己的时间戳
        （= ``t_host - tau``，模拟两个时钟的常量偏移），真实系统里两者
        是各自独立的计数器。
        """
        n = int(round((t_end - t_start) * rate))
        t_host = t_start + np.arange(n) / rate
        gyro = np.zeros((n, 3))
        accel = np.zeros((n, 3))
        for i, t in enumerate(t_host):
            gyro[i], accel[i], _ = self.imu_sample(t, rng=rng, with_noise=with_noise)
        # IMU 自己记的时间戳：t_imu = t_host - tau
        return t_host, t_host - self.tau, gyro, accel

    def make_camera_stream(self, t_start, t_end, rate=20.0, board_spec=None,
                           supersample=3, noise_sigma=0.0, rng=None):
        """渲染相机序列，返回 ``(t_host, images, rvec_true, tvec_true)``。"""
        n = int(round((t_end - t_start) * rate))
        times = t_start + np.arange(n) / rate
        imgs, rvs, tvs = [], [], []
        for t in times:
            rvec, tvec = self.board_pose_in_camera(t)
            rvs.append(rvec)
            tvs.append(tvec)
            if board_spec is not None:
                imgs.append(render_board_view(
                    board_spec, self.K, self.dist, rvec, tvec, self.image_size,
                    supersample=supersample, noise_sigma=noise_sigma, rng=rng))
        return times, imgs, np.asarray(rvs), np.asarray(tvs)


# ══════════════════════════════════════════════════════════════
#  小工具（打印真值，供自检脚本对照）
# ══════════════════════════════════════════════════════════════
def describe(rig: SimRig) -> str:
    R = rig.R_IC
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll, pitch, yaw = np.arctan2(-R[1, 2], R[1, 1]), np.pi / 2, 0.0
    e = np.rad2deg([roll, pitch, yaw])
    return (f"真值 T_IC: 欧拉角(ZYX) roll={e[0]:+.2f}° pitch={e[1]:+.2f}° "
            f"yaw={e[2]:+.2f}°\n"
            f"          t_IC = {np.array2string(rig.t_IC, precision=4)} m "
            f"(|t|={np.linalg.norm(rig.t_IC)*100:.2f} cm)\n"
            f"          杠杆臂 r = t_CI = {np.array2string(rig.r_lever, precision=4)} m\n"
            f"真值时钟偏移 tau = {rig.tau*1000:.2f} ms")
