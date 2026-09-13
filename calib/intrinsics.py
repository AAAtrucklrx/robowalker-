#!/usr/bin/env python3
"""C1 · 相机内参标定（pinhole + Brown-Conrady 畸变）。

模型
------------------------------------------------------------------
    u = fx * X/Z + cx        v = fy * Y/Z + cy
    x_d = x(1 + k1 r^2 + k2 r^4 + k3 r^6) + 2 p1 x y + p2 (r^2 + 2 x^2)

输出 ``fx, fy, cx, cy`` 与畸变 ``[k1, k2, p1, p2, k3]``（OpenCV ``plumb_bob``）。

关于"验证"的一个关键设计
------------------------------------------------------------------
**不要用标定用的那批图去算误差**——那叫自证，结果必然偏乐观。
本模块强制把数据切成 train / val：

* train 上跑 ``calibrateCamera``；
* val 上**只用标定出来的 K、dist**，靠 ``solvePnP`` 重新估每张图的位姿，
  再算重投影误差。

val 的误差才是"这组内参在新数据上好不好用"的诚实答案，
也是任务书第 4 节要的"重投影误差"。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from target_detect import BoardSpec, Detection, detect


@dataclass
class Intrinsics:
    K: np.ndarray                   # 3x3
    dist: np.ndarray                # (5,) k1 k2 p1 p2 k3
    image_size: tuple[int, int]     # (w, h)
    model: str = "pinhole"
    rms: float = float("nan")       # 标定残差（train 集）

    @property
    def fx(self):
        return float(self.K[0, 0])

    @property
    def fy(self):
        return float(self.K[1, 1])

    @property
    def cx(self):
        return float(self.K[0, 2])

    @property
    def cy(self):
        return float(self.K[1, 2])

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "intrinsics": {"fx": self.fx, "fy": self.fy,
                           "cx": self.cx, "cy": self.cy},
            "distortion": {"model": "plumb_bob",
                           "coefficients": [float(v) for v in self.dist]},
            "reprojection_error_rms": float(self.rms),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Intrinsics":
        i = d["intrinsics"]
        K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]],
                     dtype=float)
        dist = np.array(d["distortion"]["coefficients"], dtype=float)
        return cls(K, dist, tuple(d["image_size"]), d.get("model", "pinhole"),
                   d.get("reprojection_error_rms", float("nan")))

    def sanity_check(self) -> list[str]:
        """把"数值是否合理"的检查集中在这里，报告直接引用。

        这些阈值不是法律，是经验区间；偏离太多通常意味着采集动作有问题。
        """
        w, h = self.image_size
        out = []
        if abs(self.cx - w / 2) > 0.2 * w:
            out.append(f"cx={self.cx:.1f} 离图像中心 {w/2:.1f} 太远（>20% 宽）")
        if abs(self.cy - h / 2) > 0.2 * h:
            out.append(f"cy={self.cy:.1f} 离图像中心 {h/2:.1f} 太远（>20% 高）")
        if self.fx <= 0 or self.fy <= 0:
            out.append(f"fx/fy 非正：{self.fx:.1f}, {self.fy:.1f}")
        if self.fx and abs(self.fx - self.fy) / self.fx > 0.05:
            out.append(f"fx 与 fy 相差 {abs(self.fx-self.fy)/self.fx*100:.1f}% > 5%")
        if len(self.dist) >= 1 and not (-0.5 < self.dist[0] < 0.5):
            out.append(f"k1={self.dist[0]:.4f} 超出常见范围 (-0.5, 0.5)")
        return out


# ══════════════════════════════════════════════════════════════
def split_train_val(n: int, val_ratio: float = 0.2, seed: int = 0):
    """按固定随机种子切分索引。种子固定 → 结果可复现。"""
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_ratio)))
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])


def calibrate(dets: list[Detection], image_size: tuple[int, int],
              fix_principal_point: bool = False, fix_aspect: bool = False,
              fix_k3: bool = False) -> Intrinsics:
    """在给定视图上跑 ``cv2.calibrateCamera``。"""
    obj = [d.object_points for d in dets if d.found]
    img = [d.image_points for d in dets if d.found]
    if len(obj) < 4:
        raise ValueError(f"有效视图只有 {len(obj)} 张，至少需要 4 张，建议 15 张以上")

    flags = 0
    if fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    if fix_aspect:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if fix_k3:
        flags |= cv2.CALIB_FIX_K3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8)

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj, img, image_size, None, None, flags=flags, criteria=criteria)
    return Intrinsics(K, np.asarray(dist).reshape(-1), image_size, "pinhole",
                      float(rms))


def reprojection_rms(objp: np.ndarray, imgp: np.ndarray, rvec, tvec,
                     K: np.ndarray, dist: np.ndarray) -> float:
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    err = (proj.reshape(-1, 2) - imgp.reshape(-1, 2))
    return float(np.sqrt((err ** 2).sum(axis=1).mean()))


def validate_on(intr: Intrinsics, dets: list[Detection]) -> dict:
    """在**没参与标定**的视图上评估：只用 K/dist，靠 solvePnP 重估位姿。

    这是比"train 残差"更诚实的指标。
    """
    per_view = []
    for d in dets:
        if not d.found:
            continue
        ok, rvec, tvec = cv2.solvePnP(d.object_points, d.image_points,
                                      intr.K, intr.dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        rms = reprojection_rms(d.object_points, d.image_points, rvec, tvec,
                               intr.K, intr.dist)
        per_view.append(rms)
    if not per_view:
        return {"n": 0}
    a = np.asarray(per_view)
    return {
        "n": len(a),
        "rms_px": float(np.sqrt((a ** 2).mean())),
        "mean_px": float(a.mean()),
        "max_px": float(a.max()),
        "per_view_px": [float(v) for v in a],
    }


def calibrate_from_paths(paths: list, spec: BoardSpec,
                         val_ratio: float = 0.2, seed: int = 0,
                         verbose: bool = True) -> dict:
    """完整流程：读图 → 检测 → 切分 → 标定 → 验证。"""
    import cv2 as _cv2

    dets_all, used, failed = [], [], []
    image_size = None
    for p in paths:
        gray = _cv2.imread(str(p), _cv2.IMREAD_GRAYSCALE)
        if gray is None:
            failed.append(str(p))
            continue
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        elif (gray.shape[1], gray.shape[0]) != image_size:
            raise ValueError(
                f"{p} 分辨率 {gray.shape[1]}x{gray.shape[0]} 与首张 {image_size} 不一致；"
                f"标定要求所有图同分辨率")
        d = detect(gray, spec)
        dets_all.append(d)
        (used if d.found else failed).append(str(p))
        if verbose:
            print(f"  {'✅' if d.found else '❌'} {p.name if hasattr(p,'name') else p}"
                  f"  {0 if not d.found else len(d.image_points)} 点")

    good = [d for d in dets_all if d.found]
    if len(good) < 6:
        raise ValueError(f"只有 {len(good)} 张检出标定板，无法可靠标定")

    tr, va = split_train_val(len(good), val_ratio, seed)
    train = [good[i] for i in tr]
    val = [good[i] for i in va]
    if verbose:
        print(f"\n检出 {len(good)}/{len(dets_all)} 张；"
              f"train {len(train)} / val {len(val)}（seed={seed}）")

    intr = calibrate(train, image_size)
    cal_res = _residual_on_train(intr, train)
    val_res = validate_on(intr, val)

    if verbose:
        print(f"\n标定残差 (train, {cal_res['n']} 张): RMS={cal_res['rms_px']:.4f} px")
        print(f"重投影误差 (val,   {val_res.get('n',0)} 张): "
              f"RMS={val_res.get('rms_px', float('nan')):.4f} px  "
              f"max={val_res.get('max_px', float('nan')):.4f} px")
        print(f"\nfx={intr.fx:.3f}  fy={intr.fy:.3f}  "
              f"cx={intr.cx:.3f}  cy={intr.cy:.3f}")
        print("dist =", np.array2string(intr.dist, precision=6))
        issues = intr.sanity_check()
        print("\n合理性检查:", "全部通过 ✅" if not issues else "")
        for s in issues:
            print(f"  ⚠️  {s}")

    return {
        "intrinsics": intr,
        "image_size": image_size,
        "n_detected": len(good),
        "n_failed": len(failed),
        "failed_files": failed,
        "calibration_residual": cal_res,
        "validation": val_res,
        "sanity_issues": intr.sanity_check(),
        "train_idx": tr.tolist(),
        "val_idx": va.tolist(),
    }


def _residual_on_train(intr: Intrinsics, dets: list[Detection]) -> dict:
    vals = []
    for d in dets:
        if not d.found:
            continue
        ok, rvec, tvec = cv2.solvePnP(d.object_points, d.image_points,
                                      intr.K, intr.dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            vals.append(reprojection_rms(d.object_points, d.image_points,
                                         rvec, tvec, intr.K, intr.dist))
    a = np.asarray(vals) if vals else np.array([float("nan")])
    return {"n": len(vals), "rms_px": float(np.sqrt((a ** 2).mean())),
            "mean_px": float(a.mean()), "max_px": float(a.max()),
            "per_view_px": [float(v) for v in a]}


def undistort_image(img: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """去畸变。用来做"直线变直"的肉眼检查。"""
    return cv2.undistort(img, intr.K, intr.dist)


def main() -> int:
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="C1 相机内参标定")
    ap.add_argument("--images", required=True, help="图像目录")
    ap.add_argument("--pattern", default="9x6", help="内角点数，如 9x6")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--type", default="chessboard", choices=["chessboard", "charuco"])
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="把内参存成 yaml 的路径")
    args = ap.parse_args()

    c, r = (int(v) for v in args.pattern.lower().split("x"))
    spec = BoardSpec(args.type, (c, r), args.square_mm / 1000.0)
    root = Path(args.images).expanduser()
    paths = sorted(root.glob("*.png")) + sorted(root.glob("*.jpg"))
    if not paths:
        print(f"❌ {root} 里没有图像")
        return 2
    print(f"标定板: {spec}\n图像  : {root} ({len(paths)} 张)\n")

    res = calibrate_from_paths(paths, spec, args.val_ratio, args.seed)
    if args.out:
        import yaml
        p = Path(args.out).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(res["intrinsics"].to_dict(),
                                    allow_unicode=True, sort_keys=False),
                     encoding="utf-8")
        print(f"\n已写出 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
