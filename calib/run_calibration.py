#!/usr/bin/env python3
"""主入口：**一条命令跑完 C1 → C2 → C3 → C4**，全部参数来自配置文件。

任务书第 5 节要求"输入路径、标定板参数等不应直接硬编码，应能通过配置文件、
命令行参数修改"。本文件就是这条要求的落点：所有可调项都在
``config/calibration.yaml``，命令行只做覆盖。

用法::

    # 全流程
    .venv/bin/python calib/run_calibration.py --config config/calibration.yaml

    # 临时换一组数据（不改配置）
    .venv/bin/python calib/run_calibration.py --session data/session_02

    # 只标内参 / 复用已有内参
    .venv/bin/python calib/run_calibration.py --only-intrinsics
    .venv/bin/python calib/run_calibration.py --intrinsics results/intrinsics.yaml

输出（见配置 ``output`` 段）：
    results/calibration_result.yaml / .json   标定结果（可被程序重新读回）
    results/calibration_result_<时间戳>.*     快照，用于"多次独立标定对比"
    results/report/report.md                  验证报告
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calib"))

from extrinsics import (T_from_Rt, T_IC_to_dict, ax_xb_residual,  # noqa: E402
                        invert_T, motion_consistency, solve_lever_arm,
                        solve_rotation, solve_rotation_robust)
from intrinsics import Intrinsics, calibrate_from_paths  # noqa: E402
from io_data import (build_result, load_config, load_session,  # noqa: E402
                     save_result)
from sync import (estimate_bias, estimate_offset, integrate_gyro,  # noqa: E402
                  static_segment)
from target_detect import BoardSpec, detect, solve_pnp_board  # noqa: E402
import validate as V  # noqa: E402


def log(msg=""):
    print(msg, flush=True)


def hr(title):
    log("\n" + "═" * 66)
    log(f"  {title}")
    log("═" * 66)


# ══════════════════════════════════════════════════════════════
def run(args) -> int:
    t_start = time.time()
    cfg = load_config(args.config)

    # ── 命令行覆盖 ─────────────────────────────────────────
    if args.session:
        s = Path(args.session).expanduser()
        cfg["data"]["image_dir"] = str(s / "images")
        cfg["data"]["image_timestamps"] = str(s / "image_timestamps.txt")
        cfg["data"]["imu_file"] = str(s / "imu.txt")
        cfg["data"]["_session_root"] = str(s)
    if args.pattern:
        a, b = (int(v) for v in args.pattern.lower().split("x"))
        cfg["target"].setdefault("chessboard", {})["pattern_size"] = [a, b]
        cfg["target"]["type"] = "chessboard"
    if args.square_mm:
        cfg["target"].setdefault("chessboard", {})["square_size"] = args.square_mm / 1000.0

    root = Path(cfg.get("_root", ROOT))
    def P(rel):
        p = Path(str(rel)).expanduser()
        return p if p.is_absolute() else (root / p)

    spec = BoardSpec.from_config(cfg)
    val_cfg = cfg.get("validation", {})
    out_dir = P(cfg.get("output", {}).get("dir", "results"))
    report_dir = P(cfg.get("output", {}).get("report_dir", "results/report"))
    session_root = cfg["data"].get("_session_root", str(P(cfg["data"]["image_dir"]).parent))

    log(f"配置文件: {cfg['_config_path']}")
    log(f"数据目录: {session_root}")
    log(f"标定板  : {spec}")

    # ══════════════════════════════════════════════════════
    hr("C0 · 装载数据")
    # ══════════════════════════════════════════════════════
    sess = load_session(session_root,
                        image_dir=Path(cfg["data"]["image_dir"]).name,
                        image_timestamps=Path(cfg["data"]["image_timestamps"]).name,
                        imu_file=Path(cfg["data"]["imu_file"]).name,
                        time_unit=cfg.get("time", {}).get("unit", "s"))
    for k, v in sess.summary().items():
        log(f"  {k:16s} {v}")
    if len(sess) == 0:
        log("❌ 没有图像"); return 2

    # ══════════════════════════════════════════════════════
    hr("C1 · 相机内参")
    # ══════════════════════════════════════════════════════
    if args.intrinsics:
        import yaml
        intr = Intrinsics.from_dict(yaml.safe_load(P(args.intrinsics).read_text()))
        c1 = {"intrinsics": intr, "validation": {}, "calibration_residual": {},
              "n_detected": 0, "failed_files": [], "sanity_issues": intr.sanity_check()}
        log(f"  从文件载入内参: fx={intr.fx:.2f} fy={intr.fy:.2f} "
            f"cx={intr.cx:.2f} cy={intr.cy:.2f}")
    elif cfg.get("camera", {}).get("calibrate_intrinsics", True):
        c1 = calibrate_from_paths(
            sess.image_paths, spec,
            val_ratio=0.2, seed=0, verbose=False)
        log(f"  检出 {c1['n_detected']}/{len(sess.image_paths)} 张")
        log(f"  标定残差(train) RMS = {c1['calibration_residual']['rms_px']:.4f} px")
        log(f"  重投影误差(val) RMS = {c1['validation'].get('rms_px', float('nan')):.4f} px"
            f"     {V.verdict('reprojection_rms_px', c1['validation'].get('rms_px'))}")
        for s in c1["sanity_issues"]:
            log(f"  ⚠️  {s}")
    else:
        ii = cfg["camera"]["initial_intrinsics"]
        intr = Intrinsics(
            np.array([[ii["fx"], 0, ii["cx"]], [0, ii["fy"], ii["cy"]], [0, 0, 1]]),
            np.array(ii["dist"], float), tuple(cfg["camera"]["image_size"]))
        c1 = {"intrinsics": intr, "validation": {}, "calibration_residual": {},
              "n_detected": 0, "failed_files": [], "sanity_issues": []}
        log("  按要求跳过内参标定，使用配置里的初始内参")
    intr = c1["intrinsics"]

    if args.only_intrinsics:
        res = build_result(intr.to_dict(), None,
                           {"intrinsics_val": c1["validation"]},
                           {"session": session_root, "stage": "C1-only"})
        for p in save_result(res, out_dir):
            log(f"  wrote {p}")
        return 0

    if sess.imu is None:
        log("❌ 没有 IMU 数据，无法继续 C2/C3")
        return 2

    # ══════════════════════════════════════════════════════
    hr("C1b · 逐帧 PnP（相机在世界系下的轨迹）")
    # ══════════════════════════════════════════════════════
    import cv2
    R_CB, t_CB, ts = [], [], []
    prev_R = None
    n_flip = 0
    for p, tt in zip(sess.image_paths, sess.image_ts):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        d = detect(gray, spec)
        if not d.found:
            continue
        try:
            sol = solve_pnp_board(d.object_points, d.image_points, intr.K, intr.dist,
                                  prev_R=prev_R, pattern_size=spec.pattern_size)
        except Exception:  # noqa: BLE001
            continue
        prev_R = cv2.Rodrigues(sol["rvec"])[0]
        n_flip += int(sol.get("order") == "x_reversed")
        R_CB.append(prev_R); t_CB.append(sol["tvec"].reshape(3)); ts.append(tt)
    R_CB = np.asarray(R_CB); t_CB = np.asarray(t_CB); ts = np.asarray(ts)
    log(f"  PnP 成功 {len(R_CB)}/{len(sess.image_paths)} 帧"
        f"（{n_flip} 帧用 x 反序角点约定）")
    if len(R_CB) < 30:
        log("❌ 有效帧太少（<30），无法可靠标外参。"
            "检查：标定板是否始终完整入画、运动是否够丰富")
        return 2

    # 零偏（阈值从配置读，见 imu.static 段）
    st = cfg.get("imu", {}).get("static", {})
    bias = estimate_bias(sess.imu.t, sess.imu.accel, sess.imu.gyro,
                         accel_std_max=st.get("accel_std_max", 0.05),
                         gyro_std_max=st.get("gyro_std_max", 0.02),
                         min_duration_s=st.get("min_duration", 1.0))
    if bias.get("ok"):
        gyro = sess.imu.gyro - bias["gyro_bias"]
        accel = sess.imu.accel - bias["accel_bias"]
        log(f"  静止段 {bias['n_static']} 样本（占 {bias['static_fraction']*100:.1f}%）")
        log(f"  陀螺零偏 {np.array2string(bias['gyro_bias'], precision=5)} rad/s；"
            f"|a| = {bias['accel_norm_mean']:.4f} m/s²（应≈9.81）")
    else:
        gyro, accel = sess.imu.gyro, sess.imu.accel
        log(f"  ⚠️ 跳过零偏标定：{bias.get('reason')}")

    # ══════════════════════════════════════════════════════
    hr("C2 · 时间对齐")
    # ══════════════════════════════════════════════════════
    sync = estimate_offset(ts, R_CB, sess.imu.t, gyro,
                           search_range=args.search, coarse_step=args.step,
                           verbose=True)
    tau = sync["tau"]

    # ══════════════════════════════════════════════════════
    hr("C3 · Camera-IMU 外参")
    # ══════════════════════════════════════════════════════
    R_WI_all = integrate_gyro(sess.imu.t, gyro)
    idx = [int(np.argmin(np.abs(sess.imu.t - (t - tau)))) for t in ts]
    R_WI = R_WI_all[idx]
    rot_res = solve_rotation_robust(R_WI, R_CB, verbose=True)
    R_IC = rot_res["R_IC"]

    mc = motion_consistency(R_WI, R_CB, R_IC)
    log(f"  运动一致性：均值 {mc['mean_deg']:.4f}° "
        f"{V.verdict('motion_consistency_mean_deg', mc['mean_deg'])}，"
        f"max {mc['max_deg']:.4f}°")

    log("")
    lev = solve_lever_arm(ts, R_CB, t_CB, sess.imu.t, accel, R_IC,
                          tau=tau, verbose=True)
    T_IC = T_from_Rt(R_IC, lev["t_IC_m"])
    e_dict = T_IC_to_dict(T_IC)

    # ══════════════════════════════════════════════════════
    hr("C4 · 验证")
    # ══════════════════════════════════════════════════════
    frames = list(range(len(ts)))

    def solve_fn(sub):
        """在子集上重标外参（旋转 + 平移）。

        **必须先按时间排序**：Bootstrap / 交叉验证是随机抽样的，而杠杆臂的
        B样条拟合要求时间单调。
        """
        ii = np.sort(np.asarray(sub, int))
        rr = solve_rotation(R_WI[ii], R_CB[ii], verbose=False)["R_IC"]
        lv = solve_lever_arm(ts[ii], R_CB[ii], t_CB[ii], sess.imu.t, accel,
                             rr, tau=tau, pos_sigma=lev.get("pos_sigma_used"),
                             verbose=False)
        return rr, lv["t_IC_m"]

    xval = {}
    if val_cfg.get("cross_validate_splits", 0) >= 2:
        xval = V.cross_validate(frames, int(val_cfg["cross_validate_splits"]),
                                solve_fn)
        if "error" not in xval:
            log(f"  交叉验证 {xval['n_folds']} 折：旋转标准差 "
                f"{xval['rot_std_deg']:.4f}° "
                f"{V.verdict('extrinsic_rot_std_deg', xval['rot_std_deg'])}")

    boot = V.bootstrap_extrinsics(frames, solve_fn, n_boot=args.bootstrap, seed=0)
    if "error" not in boot:
        log(f"  Bootstrap {boot['n_boot']} 次：平移标准差 "
            f"{boot['trans_std_m']*1000:.3f} mm，旋转散布 "
            f"{boot['rot_spread_deg']:.4f}°")

    # IMU 噪声（只在有静止段时做）。Allan 方差要求静态数据，且越长越可信；
    # 但 1~2 秒的静止段也足以给出量级正确的噪声密度，比用占位值强。
    imu_noise = None
    mask = static_segment(sess.imu.t, sess.imu.accel, sess.imu.gyro)
    if mask.sum() >= 800:
        try:
            imu_noise = V.estimate_imu_noise(sess.imu.t[mask], gyro[mask],
                                             accel[mask])
            log(f"  Allan 方差：静止段 {int(mask.sum())} 样本"
                f"（{mask.sum()*float(np.median(np.diff(sess.imu.t))):.1f} s）"
                f"{'  ⚠️ 偏短，噪声密度只能当量级参考' if mask.sum() < 20000 else ' ✅'}")
        except ValueError as e:
            log(f"  Allan 方差跳过：{e}")
    else:
        log(f"  Allan 方差跳过：静止段只有 {int(mask.sum())} 样本"
            f"（需 ≥800 ≈ 0.8 s。采集时开头请完全静置 2 秒以上；"
            f"想精确标噪声参数建议静置 10 分钟）")

    # ── 落盘 ───────────────────────────────────────────────
    rms = c1["validation"].get("rms_px", float("nan"))
    verdict_line = (
        f"内参重投影误差 {rms:.3f} px {V.verdict('reprojection_rms_px', rms)}；"
        f"运动一致性 {mc['mean_deg']:.3f}° "
        f"{V.verdict('motion_consistency_mean_deg', mc['mean_deg'])}；"
        f"平移外参 |t_IC| = {e_dict['translation_norm_m']*100:.2f} cm，"
        f"杠杆臂残差 {lev['residual_rms_ms2']:.4f} m/s² "
        f"{V.verdict('lever_arm_residual_ms2', lev['residual_rms_ms2'])}；"
        f"重力模长偏差 {lev['gravity_norm_error']:.4f} m/s² "
        f"{V.verdict('gravity_norm_error', lev['gravity_norm_error'])}。"
    )

    notes = [
        f"时间偏移 τ = {tau*1000:+.3f} ms（``t_host = t_imu + tau``），"
        f"角度残差 RMS {sync['residual_deg_rms']:.4f}°，"
        f"代价曲线对比度 {sync['cost_contrast']:.2f}×。",
        f"角点约定：{n_flip}/{len(R_CB)} 帧使用 x 反序"
        f"（findChessboardCornersSB 的约定）。",
        "运动一致性用的是 AX=XB 的**相对旋转**残差 —— 陀螺没有绝对 yaw 基准，"
        "用绝对姿态比会凭空多出几十度误差。",
        "平移外参由**加速度计杠杆臂效应**估计；重力向量与杠杆臂联合最小二乘，"
        "`|g|` 应等于 9.80665，是内建自检。",
        f"鲁棒估计：手眼旋转用了帧级异常剔除（{rot_res.get('n_total')} → "
        f"{rot_res.get('n_used')} 帧）；杠杆臂用了 Huber 加权 IRLS"
        f"（降权 {lev.get('n_downweighted', 0)}/{lev.get('n_samples')} 个样本）。",
    ]
    if "error" in xval:
        notes.append(f"交叉验证未完成：{xval['error']}")
    if lev.get("reliable") is False:
        notes.append("⚠️ 杠杆臂 LS 残差偏大，平移外参不可信 —— 请结合尺子实测值"
                     "并明确声明不确定度，不要直接采用优化输出。")

    result = build_result(
        intr.to_dict(), e_dict,
        {"intrinsics_val": c1["validation"],
         "calibration_residual": c1.get("calibration_residual", {}),
         "motion_consistency": mc,
         "handeye_spread_deg": rot_res["max_spread_deg"],
         "handeye_cluster": rot_res.get("cluster"),
         "robust_trim": {"n_total": rot_res.get("n_total"),
                         "n_used": rot_res.get("n_used"),
                         "history": rot_res.get("trim_history")},
         "ax_xb_residual_deg_max": float(ax_xb_residual(R_WI, R_CB, R_IC).max()),
         "cross_validation": {k: v for k, v in xval.items() if k != "per_fold_R"},
         "bootstrap": {k: v for k, v in boot.items() if k != "rot_samples"},
         "lever_arm": {k: (v.tolist() if hasattr(v, "tolist") else v)
                       for k, v in lev.items()},
         "time_offset": {**{k: v for k, v in sync.items()
                            if k not in ("grid_taus", "grid_costs")},
                         "tau_s": tau},
         "imu_noise": {k: {kk: vv for kk, vv in v.items()
                           if kk not in ("taus_s", "adev")}
                       for k, v in (imu_noise or {}).items()}},
        {"session": session_root,
         "board": str(spec),
         "n_images": len(sess.image_paths),
         "n_imu": len(sess.imu),
         "n_pnp_ok": len(R_CB),
         "verdict_line": verdict_line,
         "notes": notes,
         "elapsed_s": round(time.time() - t_start, 1)})

    hr("落盘")
    for p in save_result(result, out_dir,
                         yaml_name=cfg["output"].get("yaml_name", "calibration_result.yaml"),
                         json_name=cfg["output"].get("json_name", "calibration_result.json"),
                         keep_snapshots=cfg["output"].get("keep_snapshots", True)):
        log(f"  {p}")
    txt = V.build_report(intr.to_dict(), e_dict, result["validation"],
                         result["meta"], imu_noise)
    rp = V.write_report(txt, report_dir)
    log(f"  {rp}")

    hr("完成")
    log(f"  {verdict_line}")
    log(f"  耗时 {time.time()-t_start:.1f} s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Camera-IMU 联合标定 主入口（C1→C2→C3→C4）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/calibration.yaml")
    ap.add_argument("--session", default=None, help="覆盖配置里的数据目录")
    ap.add_argument("--intrinsics", default=None, help="已有内参 yaml，跳过 C1 标定")
    ap.add_argument("--only-intrinsics", action="store_true")
    ap.add_argument("--pattern", default=None, help="内角点数，如 9x6（覆盖配置）")
    ap.add_argument("--square-mm", type=float, default=None)
    ap.add_argument("--search", type=float, default=0.5, help="τ 搜索范围 ±秒")
    ap.add_argument("--step", type=float, default=2e-3, help="τ 搜索粗步长")
    ap.add_argument("--bootstrap", type=int, default=20)
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
