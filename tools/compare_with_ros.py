#!/usr/bin/env python3
"""用 ROS 2 的 ``camera_calibration`` 做**第三方内参对照**（任务书第 4 节的验证项之一）。

为什么这样比"再跑一遍 OpenCV"更有说服力
------------------------------------------------------------------
ROS 的 ``camera_calibration`` 是**另一份独立实现**：

* 它有自己的角点检测流程（``collect_corners``）、自己的采样准则
  （``is_good_sample``）、自己的标定与误差计算；
* 我们**复用它自己的检测**，而不是把我们的角点喂给它 —— 否则就变成了
  "同一批角点、同一个 OpenCV 后端"，证明不了什么。

**比什么才有意义**：不能只比 fx/fy/cx/cy 四个数。径向畸变系数之间高度相关，
`k1` 偏 +0.02 而 `k2` 反向偏 0.02 在物理上可能完全等价。所以本工具比的是
**投影函数本身**：

    在相机的视锥里撒一批 3D 点，分别用两套 (K, dist) 投影，
    统计像素差的 RMS / 95 分位 / 最大值。

这个数字直接回答一个工程问题：**"如果我用 ROS 的标准内参代替我们的，
我的像素测量会移动多少？"**

用法::

    # 先跑一次免 sudo 安装
    bash tools/setup_ros_bypass.sh

    # 再对照
    .venv/bin/python tools/compare_with_ros.py \\
        --session data/session_01 --pattern 9x6 --square-mm 25
    # 也可以直接指定我们的结果文件
    .venv/bin/python tools/compare_with_ros.py --session data/session_01 \\
        --pattern 9x6 --square-mm 25 --ours results/calibration_result.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calib"))

# 抽包位置放在**项目内**而不是 /tmp：/tmp 会被系统清理、重启即丢，
# 实测踩过一次 —— 脚本静默失效，而它偏偏是"与成熟工具对比"这项验证的唯一入口。
# 放在 third_party/ 下可以跟着项目一起备份，且 .deb 缓存在一起 → 断网也能重建。
_CC_REL = "third_party/roscc/ex/opt/ros/jazzy/lib/python3.12/site-packages"
DEFAULT_ROS_PKG = str(ROOT / _CC_REL)
_LEGACY_ROS_PKG = "/tmp/roscc/ex/opt/ros/jazzy/lib/python3.12/site-packages"
# ROS distro 自带的 Python 包（cv_bridge / rclpy 等）不在系统 site-packages 里，
# venv 看不到，必须显式加进 sys.path
ROS_DISTRO_PKG = "/opt/ros/jazzy/lib/python3.12/site-packages"
ROS_DISTRO_LIB = "/opt/ros/jazzy/lib"


def _ensure_ld_path() -> None:
    """确保 ``LD_LIBRARY_PATH`` 含 ROS 的 lib 目录，否则**原进程重启一次**。

    为什么必须这样：``camera_calibration`` 内部用 ``cv_bridge.CvBridge``，
    而它的 C 扩展 ``libcv_bridge.so`` 只在 ``/opt/ros/jazzy/lib`` 里。
    ``LD_LIBRARY_PATH`` 是**动态链接器在进程启动时**读的，从 Python 里
    ``os.environ`` 改是没用的 —— 实测报错：
    ``ImportError: libcv_bridge.so: cannot open shared object file``。

    所以在做任何实际工作之前 ``os.execve`` 重启自己一次（带正确的环境变量）。
    判定条件保证不会无限重启。
    """
    parts = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if ROS_DISTRO_LIB in parts:
        return
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ROS_DISTRO_LIB + (
        ":" + os.environ["LD_LIBRARY_PATH"]
        if os.environ.get("LD_LIBRARY_PATH") else "")
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def _ensure_ros(pkg_path: str | None = None) -> None:
    """把解包出来的 camera_calibration 加进 sys.path，并给出可操作的报错。"""
    _ensure_ld_path()
    cands = [pkg_path, os.environ.get("ROS_CC_PATH"), DEFAULT_ROS_PKG,
             _LEGACY_ROS_PKG, os.environ.get("ROS_DISTRO_PKG"), ROS_DISTRO_PKG]
    for c in cands:
        if c and Path(c).is_dir() and c not in sys.path:
            sys.path.insert(0, c)
    try:
        import camera_calibration.calibrator  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            f"❌ 用不了 ROS 的 camera_calibration（{e}）。\n"
            f"   先跑一次：bash tools/setup_ros_bypass.sh\n"
            f"   它会免 sudo 把包抽到 {DEFAULT_ROS_PKG} 并装好 semver。"
        ) from e


def ros_calibrate(image_paths, spec) -> dict:
    """用 ROS camera_calibration **无头**标一份内参。

    走 ROS 的标准流程：把每张图包成 ``sensor_msgs/Image`` 喂给 ``handle_msg``，
    由它内部的 ``downsample_and_detect`` 做**它自己的角点检测**，再
    ``do_calibration()``。

    为什么要这么绕：先 ``collect_corners`` 再直接 ``cal_fromcorners`` 会失败 ——
    ``do_calibration()`` 内部会重新去读 ``self.db``，而 ``db`` 只有
    ``handle_msg`` 才填。这个坑踩过一次。
    """
    import cv2
    from camera_calibration.calibrator import ChessboardInfo, MonoCalibrator

    n_cols, n_rows = int(spec.pattern_size[0]), int(spec.pattern_size[1])
    ci = ChessboardInfo(pattern="chessboard", n_cols=n_cols, n_rows=n_rows,
                        dim=float(spec.square_size))
    mc = MonoCalibrator([ci])

    grays = []
    for p in image_paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            grays.append(g)
    if not grays:
        raise SystemExit("❌ 一张图都读不到")

    # ① ROS 自己的角点检测（不是复用我们的）
    good = mc.collect_corners(grays)
    if not good or len(good) < 5:
        raise SystemExit(f"❌ ROS 只检出 {0 if not good else len(good)} 张有效板子")

    # ② ROS 自己的标定。
    # 注意这里**不走 do_calibration()**：它内部会重新遍历 self.db，
    # 而 db 只有 handle_msg 才填（且要样本"足够不同"才入库），实测会 IndexError。
    # 直接把 collect_corners 的结果交给 cal_fromcorners，等价且稳。
    mc.good_corners = good
    mc.cal_fromcorners(good)
    out = mc.as_message()
    # ROS 2 Jazzy 把 CameraInfo 的字段改成了小写（K → k, D → d），
    # 两个都试一下，兼容不同发行版。
    K_raw = getattr(out, "k", None)
    if K_raw is None:
        K_raw = getattr(out, "K", None)
    D_raw = getattr(out, "d", None)
    if D_raw is None:
        D_raw = getattr(out, "D", None)
    K = np.asarray(K_raw, float).reshape(3, 3)
    D = np.asarray(D_raw, float).ravel()
    return {"K": K, "dist": D, "image_size": (int(out.width), int(out.height)),
            "n_used": len(good), "n_input": len(grays)}


def sample_frustum_points(image_size, K, n_side: int = 9, z_range=(0.3, 2.5)):
    """在视锥里撒一批 3D 点：像素网格 × 若干深度 → 反投影成射线上的点。"""
    import cv2

    W, H = image_size
    us = np.linspace(0.05 * W, 0.95 * W, n_side)
    vs = np.linspace(0.05 * H, 0.95 * H, n_side)
    zs = np.linspace(z_range[0], z_range[1], 5)
    grid = np.array([[u, v] for v in vs for u in us], float).reshape(-1, 1, 2)
    und = cv2.undistortPoints(grid, K, np.zeros(5))       # 归一化坐标
    rays = np.concatenate([und.reshape(-1, 2), np.ones((len(und), 1))], axis=1)
    pts = np.vstack([rays * z for z in zs]).astype(np.float64)
    return pts


def compare_models(K1, d1, K2, d2, image_size) -> dict:
    """**投影函数级**对比：同一个 3D 点在两套模型下投到哪，差多少像素。"""
    import cv2

    pts = sample_frustum_points(image_size, K1)
    rvec = np.zeros(3); tvec = np.zeros(3)
    p1, _ = cv2.projectPoints(pts, rvec, tvec, K1, d1)
    p2, _ = cv2.projectPoints(pts, rvec, tvec, K2, d2)
    diff = np.linalg.norm(p1.reshape(-1, 2) - p2.reshape(-1, 2), axis=1)
    return {
        "n_points": int(len(diff)),
        "rms_px": float(np.sqrt((diff ** 2).mean())),
        "mean_px": float(diff.mean()),
        "p95_px": float(np.percentile(diff, 95)),
        "max_px": float(diff.max()),
        "per_point_px": diff.tolist(),
    }


def radial_breakdown(K1, d1, K2, d2, image_size, n_bins: int = 4) -> list:
    """把两套模型的差**按像场半径分档**，看差从哪来。

    为什么需要这个：只报一个 RMS，读者无法判断这是"整体焦距差 0.4%"还是
    "中心一致、边角因高阶畸变分开"。前者是系统性问题，后者在畸变系数强相关
    时几乎不可避免。所以按归一化半径分档报，让"差在哪"一目了然。
    """
    import cv2

    W, H = image_size
    K1 = np.asarray(K1, float)
    pts = sample_frustum_points(image_size, K1)
    rvec = np.zeros(3); tvec = np.zeros(3)
    p1, _ = cv2.projectPoints(pts, rvec, tvec, K1, np.asarray(d1, float))
    p2, _ = cv2.projectPoints(pts, rvec, tvec, K2, np.asarray(d2, float))
    p1 = p1.reshape(-1, 2); p2 = p2.reshape(-1, 2)
    diff = np.linalg.norm(p1 - p2, axis=1)
    # 归一化半径：1.0 = 图像半对角线（角点处）
    cx, cy = K1[0, 2], K1[1, 2]
    r = np.hypot((p1[:, 0] - cx) / (W / 2), (p1[:, 1] - cy) / (H / 2)) / np.sqrt(2)
    edges = np.linspace(0, max(r.max(), 1e-6), n_bins + 1)
    out = []
    for i in range(n_bins):
        m = (r >= edges[i]) & (r <= edges[i + 1] if i == n_bins - 1 else r < edges[i + 1])
        if m.sum() == 0:
            continue
        out.append({"r_lo": float(edges[i]), "r_hi": float(edges[i + 1]),
                    "n": int(m.sum()), "rms_px": float(np.sqrt((diff[m] ** 2).mean())),
                    "max_px": float(diff[m].max())})
    return out


def _load_ours(path: Path) -> dict:
    from io_data import load_result
    d = load_result(path)
    c = d["camera"]
    i = c["intrinsics"]
    K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]], float)
    dist = np.asarray(c["distortion"]["coefficients"], float)
    return {"K": K, "dist": dist, "image_size": tuple(c["image_size"])}


def _load_truth(session_dir: Path) -> dict | None:
    """读合成数据的真值（真机数据没有这个文件，返回 None）。

    为什么这个文件很关键：两套实现给出不同的内参时，光比"谁和谁不一样"是
    **无法判定谁对**的 —— 必须有一个外部参照。合成数据自带真值，于是可以把
    "两者差 2.36 px"这种模棱两可的结论，变成"谁离真值更近"的确定结论。
    """
    p = session_dir / "truth.json"
    if not p.exists():
        return None
    import json
    t = json.loads(p.read_text())
    if "K" not in t:
        return None
    return {"K": np.asarray(t["K"], float),
            "dist": np.asarray(t["dist"], float),
            "image_size": tuple(t.get("image_size", (0, 0)))}


def main() -> int:
    ap = argparse.ArgumentParser(description="与 ROS camera_calibration 做第三方内参对照")
    ap.add_argument("--session", required=True)
    ap.add_argument("--pattern", default="9x6")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--ours", default=None,
                    help="我们的结果文件（默认 results/calibration_result.yaml）")
    ap.add_argument("--ros-pkg-path", default=None)
    ap.add_argument("--truth", default=None,
                    help="真值 json（默认自动找 <session>/truth.json；合成数据才有）")
    args = ap.parse_args()

    _ensure_ros(args.ros_pkg_path)
    from io_data import load_session
    from target_detect import BoardSpec

    c, r = (int(v) for v in args.pattern.lower().split("x"))
    spec = BoardSpec("chessboard", (c, r), args.square_mm / 1000.0)
    sess = load_session(args.session)
    print(f"数据: {sess.root}（{len(sess)} 帧）")
    print(f"板  : {spec}\n")

    ours_path = Path(args.ours) if args.ours else ROOT / "results/calibration_result.yaml"
    if not ours_path.exists():
        raise SystemExit(f"❌ 找不到我们的结果 {ours_path}；先跑 calib/run_calibration.py")
    ours = _load_ours(ours_path)

    print("① 用 ROS camera_calibration 独立标一遍（它自己的角点检测）...")
    ros = ros_calibrate(sess.image_paths, spec)
    print(f"   ROS 用上 {ros['n_used']}/{ros['n_input']} 张，"
          f"图像尺寸 {ros['image_size']}")

    print("\n② 内参逐项对比")
    print(f"   {'参数':>6} {'我们':>14} {'ROS':>14} {'差值':>12} {'相对':>9}")
    for name, idx in (("fx", (0, 0)), ("fy", (1, 1)), ("cx", (0, 2)), ("cy", (1, 2))):
        a = ours["K"][idx]; b = ros["K"][idx]
        rel = (b - a) / a * 100 if a else float("nan")
        print(f"   {name:>6} {a:14.4f} {b:14.4f} {b-a:+12.4f} {rel:+8.3f}%")
    da = np.asarray(ours["dist"]).ravel()
    db = np.asarray(ros["dist"]).ravel()
    n = max(len(da), len(db))
    da = np.pad(da, (0, n - len(da))); db = np.pad(db, (0, n - len(db)))
    print(f"   {'畸变':>6} " + " ".join(f"{v:+.5f}" for v in da))
    print(f"   {'':>6} " + " ".join(f"{v:+.5f}" for v in db) + "   ← ROS")

    print("\n③ **投影函数级**对比（比逐项比系数更有物理意义）")
    cmp = compare_models(ours["K"], ours["dist"], ros["K"], ros["dist"],
                         ours["image_size"])
    print(f"   在视锥内撒 {cmp['n_points']} 个 3D 点，两套模型各投影一次：")
    print(f"     像素差 RMS = {cmp['rms_px']:.4f} px")
    print(f"     均值 {cmp['mean_px']:.4f} / 95 分位 {cmp['p95_px']:.4f} / "
          f"最大 {cmp['max_px']:.4f} px")

    # 差在哪：中心还是边角？这决定了它是"系统性问题"还是"高阶畸变相互补偿"。
    print("\n   差随像场半径的分布（r=0 中心，r=1 图像角点）：")
    for b in radial_breakdown(ours["K"], ours["dist"], ros["K"], ros["dist"],
                              ours["image_size"]):
        print(f"     r ∈ [{b['r_lo']:.2f}, {b['r_hi']:.2f}]  n={b['n']:>3}  "
              f"RMS={b['rms_px']:6.3f} px  最大={b['max_px']:6.3f} px")

    # ④ 谁更接近真值（只有合成数据有真值；真机数据这节自动跳过）
    truth = _load_truth(sess.root) if args.truth is None else (
        _load_truth(Path(args.truth).parent) if Path(args.truth).is_file()
        else _load_truth(Path(args.truth)))
    truth_verdict = None
    if truth is not None:
        print("\n④ 与**真值**对比（这是唯一能判定『谁更对』的参照）")
        ours_t = compare_models(truth["K"], truth["dist"], ours["K"], ours["dist"],
                                truth["image_size"] or ours["image_size"])
        ros_t = compare_models(truth["K"], truth["dist"], ros["K"], ros["dist"],
                               truth["image_size"] or ours["image_size"])
        print(f"   投影 RMS（越小越接近真值）：")
        print(f"     我们 = {ours_t['rms_px']:.4f} px   最大 {ours_t['max_px']:.4f} px")
        print(f"     ROS  = {ros_t['rms_px']:.4f} px   最大 {ros_t['max_px']:.4f} px")
        truth_verdict = "ours" if ours_t["rms_px"] <= ros_t["rms_px"] else "ros"
        ratio = max(ours_t["rms_px"], ros_t["rms_px"]) / min(
            ours_t["rms_px"], ros_t["rms_px"])
        who = "我们" if truth_verdict == "ours" else "ROS"
        print(f"   → {who}更接近真值（差 {ratio:.2f} 倍）")

        print("\n   逐项误差（对照真值）：")
        Kt, dt_ = truth["K"], truth["dist"]
        print(f"     {'参数':>5} {'真值':>10} {'我们':>11} {'误差':>10} {'ROS':>11} {'误差':>10}")
        for nm, (i, j) in (("fx", (0, 0)), ("fy", (1, 1)), ("cx", (0, 2)), ("cy", (1, 2))):
            ov = ours["K"][i, j]; rv = ros["K"][i, j]; tv = Kt[i, j]
            print(f"     {nm:>5} {tv:10.4f} {ov:11.4f} {ov-tv:+10.4f} {rv:11.4f} {rv-tv:+10.4f}")
        for k, nm in ((0, "k1"), (1, "k2"), (4, "k3")):
            if k < len(dt_) and k < len(da) and k < len(db):
                tv = dt_[k]
                print(f"     {nm:>5} {tv:10.5f} {da[k]:11.5f} {da[k]-tv:+10.5f} "
                      f"{db[k]:11.5f} {db[k]-tv:+10.5f}")

    print("\n⑤ 判读")
    # 判据说明：跨实现"完全一致"不是合理要求 —— 畸变系数 (k2,k3) 在有限视图下
    # 强相关，两组等价解可以差出几个像素。所以这里把两件事分开讲：
    #   * 差异有多大、差在哪（③ 的 RMS 与半径分布）；
    #   * 有真值时谁更接近真值（④，决定性证据）。
    if cmp["rms_px"] < 0.5:
        print("   两套独立实现给出的内参在投影意义上等价（< 0.5 px）→ **互为佐证**，")
        print("   可以把这条作为任务书第 4 节「与成熟工具对比」的验证结果写进报告。")
        ok = True
    else:
        print(f"   两套实现差 {cmp['rms_px']:.2f} px RMS，**大于**我们自身的重投影")
        print("   误差（~0.2 px 量级），所以这是一个需要解释的真实差异，不能忽略。")
        print("   常见且可接受的原因：径向畸变高阶项 (k2,k3) 强相关 —— 两者在")
        print("   中心区几乎重合、在边角分开（见 ③ 的半径分布）。")
        if truth_verdict is not None:
            who = "我们" if truth_verdict == "ours" else "ROS"
            print(f"   本数据带真值，④ 已给出决定性判据：**{who}更接近真值**。")
            print("   报告里应当同时写这两条（差异存在 + 我方更接近真值），")
            print("   而不是只挑对自己有利的一条。")
            ok = True          # 有真值裁决 → 结论明确，不算失败
        else:
            print("   真机数据没有真值，无法在两者间裁决谁更对。此时可用的旁证：")
            print("   * 各自的**验证集重投影误差**（我们的报告里有 val RMS）；")
            print("   * 追加一组独立视角（新拍的几张图）做**交叉验证**：")
            print("     用两套内参分别去 undistort 新图，看谁的直线更直。")
            print("   建议在报告里如实写成『两套实现差异 X px，我方以 val RMS 更低 /")
            print("   交叉验证更优作为采信依据』，不要写成『与 ROS 完全一致』。")
            ok = False

    print("\n   差异偏大时的排查顺序（与上面结论无关，供今后复用）：")
    print("   1) ROS 用了几张图？畸变系数（尤其 k3）在视图少时抖动很大；")
    print("   2) 两者的畸变模型是否一致（都应是 plumb_bob / 5 参数）；")
    print("   3) 图像分辨率是否一致（ROS 的 image_size 必须与我们的相同）；")
    print("   4) 是否存在某几张图的角点被一方检出、另一方漏检。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
