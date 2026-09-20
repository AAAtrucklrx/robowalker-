#!/usr/bin/env python3
"""内参标定「外部数据集基准测试」—— 用公开数据反复验证我们的实现。

为什么要做这个
==================================================================
我们缺**高质量的真实相机-IMU 采集数据**（受光照、支架、人手限制）。但
**内参标定（C1）** 完全可以拿公开数据集来验证，因为：

* 内参只依赖图像 + 已知棋盘格，**不需要 IMU**；
* 而且内参**与方格尺寸无关**（物点整体缩放不改变张正友法的归一化约束）
  —— 所以不用知道数据集印了多大的格子。

于是可以攒一批公开数据集，反复跑、横向对比。这比"自己采一次数据自证"
有力得多，因为**数据来源独立于我们的代码**。

判据（三条都是**自洽**的，不依赖任何外部参考值）
==================================================================
1. **重投影残差**：用该模型的 K/dist 反解每张图的位姿，看投影能不能拟合。
   越低越好。注意 fx 和畸变会互相补偿，所以**单看残差不够**，要配 2、3。
2. **畸变物理合理性**：径向畸变倍率随半径应平缓变化。如果在拟合区域之外
   迅速爆炸（几十倍），就是**高阶项过拟合**。
3. **去畸变后直线直不直**：棋盘格的行/列在真实世界是直线，去畸变后也该是
   直线。**这条最硬** —— 它不关心参数长什么样，只看结果对不对。

用法::

    # 跑所有已登记的数据集
    .venv/bin/python tools/bench_intrinsics.py

    # 只跑某个
    .venv/bin/python tools/bench_intrinsics.py --only opencv_left

    # 列出已登记的数据集
    .venv/bin/python tools/bench_intrinsics.py --list

    # 临时加一个（目录 + 内角点规格）
    .venv/bin/python tools/bench_intrinsics.py --add /path/to/imgs --pattern 9x6
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calib"))
sys.path.insert(0, str(ROOT / "tools"))

# ── 已登记的数据集 ────────────────────────────────────────────────
# 每加一个数据集，往这里加一条即可。
DATASETS: list[dict] = [
    {
        "name": "opencv_left",
        "dir": "data/opencv_samples",
        "glob": "left*.jpg",
        "pattern": (9, 6),          # 内角点
        "square_mm": 25.0,          # 只影响平移，不影响内参
        "note": "OpenCV 官方样例（left01-14，13 张，640x480）",
    },
    {
        "name": "opencv_right",
        "dir": "data/opencv_samples",
        "glob": "right*.jpg",
        "pattern": (9, 6),
        "square_mm": 25.0,
        "note": "OpenCV 官方样例右相机（独立的一组图）",
    },
    {
        "name": "opencv_all",
        "dir": "data/opencv_samples",
        "glob": "*.jpg",
        "pattern": (9, 6),
        "square_mm": 25.0,
        "note": "左右合并（注意：两台不同相机，内参不该混标，仅作对照）",
    },
]

# 结果存这里（可累积）
RECORD = ROOT / "results" / "bench_intrinsics.json"


# ══════════════════════════════════════════════════════════════
#  三条判据
# ══════════════════════════════════════════════════════════════
def judge_reprojection(K, dist, dets) -> dict:
    """用给定 K/dist 反解每张图位姿，看重投影残差（px，越低越好）。"""
    import cv2

    errs = []
    for d in dets:
        ok, rvec, tvec = cv2.solvePnP(d.object_points, d.image_points, K, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(d.object_points, rvec, tvec, K, dist)
        e = proj.reshape(-1, 2) - d.image_points.reshape(-1, 2)
        errs.append(float(np.sqrt((e ** 2).sum(axis=1).mean())))
    if not errs:
        return {"mean_px": float("nan"), "max_px": float("nan"), "n": 0}
    return {"mean_px": float(np.mean(errs)), "max_px": float(np.max(errs)),
            "n": len(errs)}


def judge_distortion_sanity(K, dist, factors=(1.0, 2.0, 4.0, 8.0)) -> dict:
    """径向畸变倍率随半径的变化（越平缓越物理）。

    倍率 = 1 + k1·r² + k2·r⁴ + k3·r⁶。
    在图像边角（r²=r_max²）处算，然后按 factors 放大半径看外推行为。
    如果外推几倍就爆到几十倍，说明高阶项在拟合噪声。
    """
    k1, k2, k3 = float(dist[0]), float(dist[1]), float(dist[4])
    r2_max = (K[0, 2] / K[0, 0]) ** 2 + (K[1, 2] / K[1, 1]) ** 2
    vals = []
    for f in factors:
        r2 = r2_max * f
        vals.append(float(1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3))
    return {"factors": list(factors), "scale": vals,
            "worst": float(max(abs(v) for v in vals))}


def judge_undistort_straightness(K, dist, dets, pattern) -> dict:
    """**最硬的判据**：去畸变后，棋盘格的行/列应该还是直线。

    对每一行和每一列拟合直线，算点到直线的平均偏差。
    ``cv2.undistortPoints`` 返回的是**归一化坐标**，乘 fx 换成像素约数。

    ⚠️ ``pattern`` 必须传真实的内角点规格。早先想"自动推断列数"（找第一个
    能整除总数的 c），结果 54 个点被猜成 3 列 —— 把"连续 3 个点"当成一条
    直线去拟合，残差毫无意义（两个模型都得到 ~8.5，看起来"一样差"）。
    这种**静默的错误判据**比报错更危险，所以这里必须显式传入。
    """
    import cv2

    res = []
    for d in dets:
        pts = d.image_points.reshape(-1, 2).astype(np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, np.asarray(K, float),
                                  np.asarray(dist, float)).reshape(-1, 2)
        n = len(und)
        # pattern 必填。**故意不做"自动猜列数"** —— 猜错会得到一个静默的
        # 错误判据（实测 54 个点被猜成 3 列，两个模型都得到无意义的 ~8.5，
        # 看起来"一样差"）。宁可报错，也不要给出看起来合理的错数字。
        cols = int(pattern[0])
        if n % cols:
            raise ValueError(
                f"内角点数 {n} 不能被 pattern 宽度 {cols} 整除 —— "
                f"pattern 传错了吗？")
        g = und.reshape(n // cols, cols, 2)
        for line in list(g) + list(g.transpose(1, 0, 2)):
            x, y = line[:, 0], line[:, 1]
            if np.ptp(x) < 1e-9 and np.ptp(y) < 1e-9:
                continue
            # 用总体最小二乘（对方向不敏感）
            c0 = line.mean(axis=0)
            _, _, vt = np.linalg.svd(line - c0)
            nrm = vt[1]
            res.append(float(np.abs((line - c0) @ nrm).mean()))
    if not res:
        return {"mean": float("nan"), "n_lines": 0}
    return {"mean": float(np.mean(res)), "n_lines": len(res)}


# ══════════════════════════════════════════════════════════════
#  跑一个数据集
# ══════════════════════════════════════════════════════════════
def run_dataset(ds: dict, with_ros: bool = True, verbose: bool = True) -> dict:
    from target_detect import BoardSpec, detect
    from intrinsics import calibrate_from_paths

    files = sorted(glob.glob(str(ROOT / ds["dir"] / ds["glob"])))
    if not files:
        return {"name": ds["name"], "error": f"没找到图：{ds['dir']}/{ds['glob']}"}

    spec = BoardSpec("chessboard", tuple(ds["pattern"]),
                     ds.get("square_mm", 25.0) / 1000.0)

    out: dict = {"name": ds["name"], "note": ds.get("note", ""),
                 "n_files": len(files), "pattern": list(ds["pattern"])}

    # ── 我们的实现 ────────────────────────────────────────────
    c1 = calibrate_from_paths(files, spec, val_ratio=0.25, seed=0, verbose=False)
    intr = c1["intrinsics"]
    out["ours"] = {
        "fx": float(intr.fx), "fy": float(intr.fy),
        "cx": float(intr.cx), "cy": float(intr.cy),
        "dist": [float(v) for v in intr.dist],
        "n_detected": int(c1["n_detected"]),
        "train_rms_px": float(c1["calibration_residual"]["rms_px"]),
        "val_rms_px": float(c1["validation"].get("rms_px", float("nan"))),
        "sanity_issues": c1["sanity_issues"],
    }

    # 三条判据要在**同一批检测结果**上比，所以重新检测一遍（关掉经典回退）
    import cv2

    dets = []
    for f in files:
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        d = detect(g, spec, allow_classic=False)
        if d.found:
            dets.append(d)
    out["n_detected_bench"] = len(dets)

    K_o = intr.K
    d_o = np.asarray(intr.dist, float)
    out["ours"]["reproj"] = judge_reprojection(K_o, d_o, dets)
    out["ours"]["dist_sanity"] = judge_distortion_sanity(K_o, d_o)
    out["ours"]["straightness"] = judge_undistort_straightness(
        K_o, d_o, dets, ds["pattern"])
    # 归一化直线残差 → 换成"约多少像素"
    out["ours"]["straightness_px"] = (
        out["ours"]["straightness"]["mean"] * float(intr.fx))

    # ── ROS 第三方实现 ────────────────────────────────────────
    if with_ros:
        try:
            import compare_with_ros as C

            C._ensure_ros(None)
            ros = C.ros_calibrate(files, spec)
            K_r, d_r = ros["K"], np.asarray(ros["dist"], float)
            out["ros"] = {
                "fx": float(K_r[0, 0]), "fy": float(K_r[1, 1]),
                "cx": float(K_r[0, 2]), "cy": float(K_r[1, 2]),
                "dist": [float(v) for v in d_r],
                "n_used": int(ros["n_used"]), "n_input": int(ros["n_input"]),
                "reproj": judge_reprojection(K_r, d_r, dets),
                "dist_sanity": judge_distortion_sanity(K_r, d_r),
                "straightness": judge_undistort_straightness(
                    K_r, d_r, dets, ds["pattern"]),
            }
            out["ros"]["straightness_px"] = (
                out["ros"]["straightness"]["mean"] * float(K_r[0, 0]))
        except Exception as e:                              # noqa: BLE001
            out["ros"] = {"error": f"{type(e).__name__}: {e}"}

    if verbose:
        print_dataset(out)
    return out


def print_dataset(o: dict) -> None:
    print("\n" + "=" * 74)
    print(f"  数据集: {o['name']}   {o.get('note','')}")
    print("=" * 74)
    if "error" in o:
        print("  ❌", o["error"])
        return
    print(f"  图 {o['n_files']} 张，检出 {o.get('n_detected_bench','?')} 张，"
          f"内角点 {o['pattern'][0]}x{o['pattern'][1]}")

    a, b = o["ours"], o.get("ros", {})
    print(f"\n  {'':<22}{'我们':>14}{'ROS':>14}")
    def row(label, av, bv, fmt="{:.4f}"):
        avs = fmt.format(av) if isinstance(av, (int, float)) else str(av)
        bvs = fmt.format(bv) if isinstance(bv, (int, float)) else str(bv)
        print(f"  {label:<22}{avs:>14}{bvs:>14}")
    row("fx", a["fx"], b.get("fx", float("nan")))
    row("fy", a["fy"], b.get("fy", float("nan")))
    row("cx", a["cx"], b.get("cx", float("nan")))
    row("cy", a["cy"], b.get("cy", float("nan")))
    print(f"  {'畸变':<22}{str([round(v,4) for v in a['dist']]):>14}"
          f"{str([round(v,4) for v in b.get('dist',[])]):>14}")
    print()
    row("① 重投影残差 mean px", a["reproj"]["mean_px"],
        b.get("reproj", {}).get("mean_px", float("nan")))
    row("② 畸变外推最坏倍率", a["dist_sanity"]["worst"],
        b.get("dist_sanity", {}).get("worst", float("nan")), "{:.2f}")
    row("③ 去畸变直线残差 px", a["straightness_px"],
        b.get("straightness_px", float("nan")), "{:.4f}")
    va = a.get("val_rms_px", float("nan"))
    print(f"\n  我们验证集重投影: {va:.4f} px"
          f"   自检: {a['sanity_issues'] or '无'}")


# ══════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description="内参标定外部数据集基准测试")
    ap.add_argument("--only", default=None, help="只跑名字含该子串的数据集")
    ap.add_argument("--list", action="store_true", help="列出已登记数据集")
    ap.add_argument("--no-ros", action="store_true", help="跳过 ROS 对照")
    ap.add_argument("--add", default=None, help="临时加一个图片目录")
    ap.add_argument("--pattern", default="9x6")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--save", action="store_true", help="结果写入 results/")
    args = ap.parse_args()

    if args.list:
        print("已登记的数据集：")
        for d in DATASETS:
            print(f"  {d['name']:<18} {d['dir']}/{d['glob']:<16} "
                  f"{d['pattern'][0]}x{d['pattern'][1]}   {d.get('note','')}")
        return 0

    dss = list(DATASETS)
    if args.add:
        c, r = (int(v) for v in args.pattern.lower().split("x"))
        p = Path(args.add)
        dss = [{"name": p.name, "dir": str(p.parent), "glob": f"{p.name}/*",
                "pattern": (c, r), "square_mm": args.square_mm,
                "note": f"临时加入：{p}"}]
    if args.only:
        dss = [d for d in dss if args.only in d["name"]]
    if not dss:
        print("没有匹配的数据集"); return 1

    results = []
    for ds in dss:
        try:
            results.append(run_dataset(ds, with_ros=not args.no_ros))
        except Exception as e:                              # noqa: BLE001
            import traceback
            traceback.print_exc()
            results.append({"name": ds["name"], "error": str(e)})

    # ── 汇总表 ────────────────────────────────────────────────
    ok = [r for r in results if "error" not in r]
    if len(ok) > 1:
        print("\n" + "=" * 74)
        print("  汇总（③ 去畸变直线残差，越小越好）")
        print("=" * 74)
        print(f"  {'数据集':<20}{'我们':>12}{'ROS':>12}{'谁好':>10}")
        for r in ok:
            a = r["ours"]["straightness_px"]
            b = r.get("ros", {}).get("straightness_px", float("nan"))
            who = "我们" if (np.isfinite(a) and (not np.isfinite(b) or a < b)) else "ROS"
            print(f"  {r['name']:<20}{a:>12.4f}{b:>12.4f}{who:>10}")

    if args.save:
        RECORD.parent.mkdir(parents=True, exist_ok=True)
        old = {}
        if RECORD.exists():
            try:
                old = json.loads(RECORD.read_text())
            except Exception:                               # noqa: BLE001
                old = {}
        for r in results:
            old[r["name"]] = r
        RECORD.write_text(json.dumps(old, ensure_ascii=False, indent=2))
        print(f"\n  已记录到 {RECORD.relative_to(ROOT)}"
              f"（累计 {len(old)} 个数据集）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
