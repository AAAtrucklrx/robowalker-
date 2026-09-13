#!/usr/bin/env python3
"""C2/C3 合成自检：在**真值已知**的数字孪生上跑完整管线。

覆盖三件事
------------------------------------------------------------------
1. **手眼约定校验**（最高风险项）：把真值的 ``R_WI`` / ``R_CB`` 直接喂给
   ``cv2.calibrateHandEye``，看它返回的是不是我们的 ``R_IC``。
   约定搞反不会报错、只会悄悄错，所以必须先用真值卡一遍。
2. **C2 时间对齐**：用渲染图做 PnP → 跑 ``estimate_offset`` → 看能否解回真值 τ。
3. **模拟器导数精度**：确认 ``simulate.py`` 里用中心差分算的角速度/角加速度
   足够准（否则 C3 的杠杆臂会被数值噪声毁掉）。

用法::

    .venv/bin/python tools/selftest_pipeline.py
    .venv/bin/python tools/selftest_pipeline.py --cam-rate 20 --duration 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calib"))

from simulate import SimRig, Trajectory, describe  # noqa: E402
from sync import (estimate_offset, prepare_gyro_integral,  # noqa: E402
                  integrated_angles, rot_angle)
from target_detect import BoardSpec, detect, solve_pnp_board  # noqa: E402


def hr(title):
    print("\n" + "─" * 64)
    print(title)
    print("─" * 64)


# ══════════════════════════════════════════════════════════════
def check_derivatives(rig: SimRig) -> bool:
    """验证 simulate.py 的数值导数是否足够精确。

    检验恒等式 ``Ṙ = R [ω]×``（ω 为世界系角速度），用有限差分算出的 Rdd
    与解析关系对照：``Rdd ≈ Ṙ[ω_body]× + R[ω̇_body]×``。
    更简单的独立检验：``Ṙ ≈ R(t+h)-R(t-h))/(2h)`` 与 ``R @ skew(ω_body)`` 对照。
    """
    hr("① 模拟器导数精度")
    t = 1.234
    h = 1e-5
    d = rig.traj.derivatives(t)
    R, w_W = d["R"], d["w_W"]
    Rd_num = (rig.traj.R_WC(t + h) - rig.traj.R_WC(t - h)) / (2 * h)
    w_body = R.T @ w_W
    K = np.array([[0, -w_body[2], w_body[1]],
                  [w_body[2], 0, -w_body[0]],
                  [-w_body[1], w_body[0], 0]])
    Rd_ana = R @ K
    err = np.abs(Rd_num - Rd_ana).max()
    print(f"  ‖Ṙ_num − R[ω]×‖∞ = {err:.3e}   "
          f"{'✅' if err < 1e-6 else '❌ 导数精度不足'}")
    # 角速度本身与"相邻帧旋转角/Δt"的一致性（相对误差，二阶近似本身就差这一项）
    dt = 1e-3
    ang = rot_angle(rig.traj.R_WC(t + dt) @ rig.traj.R_WC(t).T)
    rel = abs(np.linalg.norm(w_W) - ang / dt) / np.linalg.norm(w_W)
    print(f"  |ω| = {np.linalg.norm(w_W):.6f} rad/s  vs  "
          f"角度/Δt = {ang/dt:.6f} rad/s   相对差 {rel:.2e}   "
          f"{'✅' if rel < 1e-3 else '❌'}")
    return err < 1e-6


def check_handeye_convention(rig: SimRig, cam_times, R_CB, t_CB) -> bool:
    """用**真值**的 R_WI / R_CB 喂给 calibrateHandEye，验证约定与符号。

    结论应当是：``R_gripper2base = R_WI^T``（IMU 姿态在世界系下的逆），
    ``R_target2cam = R_CB``（PnP 直接给的），输出 = ``R_IC``（相机->IMU）。
    """
    hr("② 手眼标定约定校验（把真值喂进去，看解出来是不是 R_IC）")
    R_g2b, t_g2b, R_t2c, t_t2c, axes = [], [], [], [], []
    for i, t in enumerate(cam_times):
        _, _, info = rig.imu_sample(float(t), with_noise=False)
        R_WI = info["R_WI"]
        # T_gripper2base = T_I^W（IMU 系 -> 世界系），它的**旋转就是 R_WI**。
        # 这里踩过一次坑：写成 R_WI.T 会让所有方法都错几十度，且不报错。
        R_g2b.append(R_WI)
        t_g2b.append(np.zeros(3))         # 真机拿不到 IMU 的绝对位置，先给 0
        R_t2c.append(R_CB[i])             # T_target2cam = T_B^C（PnP 直接给的）
        t_t2c.append(t_CB[i].reshape(3))

    # 先验证数据本身满足 AX=XB（真值 X 的残差应当为 0）
    res_true = []
    for i in range(len(R_CB) - 1):
        A = R_g2b[i + 1].T @ R_g2b[i]
        B = R_t2c[i + 1] @ R_t2c[i].T
        res_true.append(rot_angle(A @ rig.R_IC @ (rig.R_IC @ B).T))
        ax, _ = cv2.Rodrigues(A)
        axes.append(ax.reshape(3))
    print(f"  数据自检：真值 X 的 AX=XB 旋转残差 max = "
          f"{np.rad2deg(max(res_true)):.2e}°  "
          f"{'✅ 数据满足 AX=XB' if np.rad2deg(max(res_true)) < 1e-6 else '❌'}")
    axes = np.asarray(axes)
    axes /= np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-12)
    sv = np.linalg.svd(axes, compute_uv=False)
    print(f"  相对旋转轴的分布（奇异值） = {np.array2string(sv, precision=3)}"
          f"   （需要至少 2 个非零 → 至少 2 个非平行轴）")

    methods = [("TSAI", cv2.CALIB_HAND_EYE_TSAI), ("PARK", cv2.CALIB_HAND_EYE_PARK),
               ("HORAUD", cv2.CALIB_HAND_EYE_HORAUD),
               ("ANDREFF", cv2.CALIB_HAND_EYE_ANDREFF),
               ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS)]
    errs = {}
    for name, m in methods:
        try:
            Rx, tx = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=m)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:11s} 抛异常 {e}")
            continue
        errs[name] = np.rad2deg(rot_angle(Rx.T @ rig.R_IC))
        print(f"  {name:11s} 与真值差角 = {errs[name]:9.4f}°   "
              f"{'✅' if errs[name] < 0.5 else '❌'}")

    ok = bool(errs) and max(errs.values()) < 0.5
    if not ok:
        print("  ⚠️  注意：若'数据自检'通过而所有方法都失败，说明是约定/输入问题；")
        print("     若数据自检也失败，说明 R_WI / R_CB 本身不自洽。")
    print(f"  （解出的 t 不可用：真机拿不到 IMU 的绝对位置，平移只能另想办法）")
    return ok


def check_sync(rig: SimRig, cam_times, R_CB, t_imu, gyro,
               search=0.4, step=5e-3) -> bool:
    hr("③ C2 时间对齐")
    print(f"  真值 τ = {rig.tau*1000:+.3f} ms")
    info = estimate_offset(np.asarray(cam_times), np.asarray(R_CB),
                           np.asarray(t_imu), np.asarray(gyro),
                           search_range=search, coarse_step=step, verbose=True)
    err_ms = (info["tau"] - rig.tau) * 1000
    ok = abs(err_ms) < 1.0 and info["reliable"]
    print(f"\n  τ 误差 = {err_ms:+.3f} ms   "
          f"{'✅' if abs(err_ms) < 1.0 else '❌ 偏差过大'}")
    # 换算成姿态误差：90°/s 下这个 τ 误差意味着多少度
    print(f"  折算：90°/s 转动下，该 τ 误差 → {abs(err_ms)*1e-3*90:.4f}° 姿态误差")
    return ok


def check_extrinsics(rig: SimRig, cam_times, R_CB, t_CB, t_imu, gyro, accel) -> bool:
    """C3：解 R_IC 与 t_IC，和真值比对。

    刻意走**真实管线**：陀螺先减零偏再积分（模拟静止段零偏标定），
    不用 rig 给的"完美姿态"。
    """
    from extrinsics import (motion_consistency, solve_lever_arm,
                            solve_rotation, T_IC_to_dict, T_from_Rt,
                            R_IC_to_lever)
    from sync import integrate_gyro

    hr("④ C3 旋转外参 R_IC")
    cam_times = np.asarray(cam_times, float)
    gyro_c = np.asarray(gyro) - rig.gyro_bias
    accel_c = np.asarray(accel) - rig.accel_bias

    # 陀螺积分得 R_WI。A = R_WI(2)^T R_WI(1) 对"世界系初始朝向"不变，
    # 所以初值取单位阵不影响旋转外参（但影响不了 motion_consistency，
    # 因为它也只用相对旋转）。
    R_WI_all = integrate_gyro(np.asarray(t_imu), gyro_c)
    idx = [int(np.argmin(np.abs(np.asarray(t_imu) - (t - rig.tau))))
           for t in cam_times]
    R_WI = R_WI_all[idx]

    rot_res = solve_rotation(R_WI, R_CB, verbose=True)
    R_IC = rot_res["R_IC"]
    err_deg = np.rad2deg(rot_angle(R_IC.T @ rig.R_IC))
    ok_rot = err_deg < 2.0
    print(f"  解出 R_IC 与真值差角 = {err_deg:.4f}°   "
          f"{'✅' if ok_rot else '❌'}（验收参考 < 1~2°）")

    mc = motion_consistency(R_WI, R_CB, R_IC)
    print(f"  运动一致性（相对旋转残差）：均值 {mc['mean_deg']:.4f}°，"
          f"max {mc['max_deg']:.4f}°")

    hr("⑤ C3 平移外参 t_IC（杠杆臂）")
    print(f"  真值 r = {np.array2string(rig.r_lever, precision=4)} m，"
          f"t_IC = {np.array2string(rig.t_IC, precision=4)} m")
    lev = solve_lever_arm(cam_times, R_CB, t_CB, np.asarray(t_imu), accel_c,
                          R_IC, tau=rig.tau, verbose=True)
    r_err = np.linalg.norm(lev["r_lever_m"] - rig.r_lever)
    ok_lev = r_err < 0.02                      # 2 cm
    print(f"  杠杆臂误差 = {r_err*100:.2f} cm   "
          f"{'✅' if ok_lev else '❌'}（容差 2 cm）")

    d = T_IC_to_dict(T_from_Rt(R_IC, lev["t_IC_m"]))
    print(f"  解出 t_IC = {np.array2string(np.asarray(d['translation_m']), precision=4)} m"
          f"  (|t|={d['translation_norm_m']*100:.2f} cm)")
    return ok_rot and ok_lev


# ══════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description="C2/C3 合成自检")
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--cam-rate", type=float, default=20.0)
    ap.add_argument("--imu-rate", type=float, default=1000.0)
    ap.add_argument("--supersample", type=int, default=4)
    ap.add_argument("--search", type=float, default=0.4)
    ap.add_argument("--step", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # 轨迹：相机离板远一点（板在画面里更小），转角幅度小但**频率高**——
    # 这样既能保证整块板不出画，又能拿到足够的角速度给 τ 估计用。
    # 真实采集的取舍完全一样：板要在画面里，同时身体要转得够快。
    traj = Trajectory(rot_amp=np.array([0.20, 0.15, 0.26]),
                      rot_freq=np.array([2.6, 3.6, 1.5]),
                      pos_amp=np.array([0.05, 0.04, 0.03]),
                      center=np.array([0.0, 0.0, 1.30]))
    rig = SimRig(traj=traj)
    spec = BoardSpec("chessboard", (9, 6), 0.025)

    print("=" * 64)
    print("Camera-IMU 标定 · C2/C3 合成自检")
    print("=" * 64)
    print(describe(rig))

    t0, t1 = 0.0, args.duration
    # IMU 时间范围要比相机宽，才能让 τ 搜索时窗口仍落在 IMU 数据内
    pad = args.search + 0.2
    print(f"\n生成数据：相机 {args.cam_rate} Hz / IMU {args.imu_rate} Hz / "
          f"{args.duration} s ...")
    t_imu_host, t_imu, gyro, accel = rig.make_imu_stream(
        t0 - pad, t1 + pad, rate=args.imu_rate, rng=rng, with_noise=True)
    cam_times, imgs, rvec_true, tvec_true = rig.make_camera_stream(
        t0, t1, rate=args.cam_rate, board_spec=spec,
        supersample=args.supersample, noise_sigma=0.0, rng=rng)
    print(f"  IMU  {len(t_imu)} 样本，相机 {len(imgs)} 帧")

    # ── 检测 + PnP（假设 C1 已完成，这里用真值内参）──────────
    # 注意：必须用 solve_pnp_board 而不是裸 solvePnP —— 平面目标有两解，
    # 裸 solvePnP 会有一半概率返回翻转解，让 C2/C3 全崩（见该函数 docstring）。
    print("\n逐帧检测 + PnP ...")
    R_CB, t_CB, ok_times = [], [], []
    prev_R = None
    ambig = 0
    for k, img in enumerate(imgs):
        d = detect(img, spec)
        if not d.found:
            continue
        try:
            sol = solve_pnp_board(d.object_points, d.image_points,
                                  rig.K, rig.dist, prev_R=prev_R)
        except Exception:  # noqa: BLE001
            continue
        R = cv2.Rodrigues(sol["rvec"])[0]
        prev_R = R
        ambig += int(sol["ambiguous"])
        R_CB.append(R)
        t_CB.append(sol["tvec"].reshape(3))
        ok_times.append(cam_times[k])
    print(f"  PnP 成功 {len(R_CB)}/{len(imgs)} 帧（其中 {ambig} 帧存在二义性，已消解）")
    if len(R_CB) < 10:
        print("❌ PnP 成功帧太少，检查轨迹是否让标定板出画了")
        return 1

    results = {
        "① 模拟器导数": check_derivatives(rig),
        "② 手眼约定": check_handeye_convention(rig, ok_times, R_CB, t_CB),
        "③ C2 时间对齐": check_sync(rig, ok_times, R_CB, t_imu, gyro,
                                    args.search, args.step),
    }
    results["④⑤ C3 外参"] = check_extrinsics(rig, ok_times, R_CB, t_CB,
                                             t_imu, gyro, accel)

    hr("总结")
    for k, v in results.items():
        print(f"  {k:16s} {'✅ 通过' if v else '❌ 未通过'}")
    allok = all(results.values())
    print("─" * 64)
    print("✅ C2 链路可信" if allok else "❌ 有未通过项")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
