#!/usr/bin/env python3
"""数据集与标定结果的读写（格式定义见 README 第 5、6 节）。

设计原则：**所有路径与列定义都来自配置文件**，核心代码不硬编码任何路径。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


# ══════════════════════════════════════════════════════════════
#  数据结构
# ══════════════════════════════════════════════════════════════
@dataclass
class ImuData:
    """IMU 序列。单位内部统一为 SI：秒 / rad/s / m/s^2。"""

    t: np.ndarray                 # (N,) 秒
    gyro: np.ndarray              # (N, 3) rad/s
    accel: np.ndarray             # (N, 3) m/s^2

    def __len__(self):
        return len(self.t)

    def slice_time(self, t0: float, t1: float) -> "ImuData":
        m = (self.t >= t0) & (self.t <= t1)
        return ImuData(self.t[m], self.gyro[m], self.accel[m])


@dataclass
class Session:
    """一次采集的全部数据。"""

    root: Path
    image_paths: list[Path] = field(default_factory=list)
    image_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    imu: ImuData | None = None

    def __len__(self):
        return len(self.image_paths)

    def summary(self) -> dict:
        d = {"root": str(self.root), "n_images": len(self.image_paths)}
        if len(self.image_ts):
            span = float(self.image_ts[-1] - self.image_ts[0])
            d["image_span_s"] = span
            d["image_rate_hz"] = (len(self.image_ts) - 1) / span if span > 0 else 0.0
        if self.imu is not None:
            d["n_imu"] = len(self.imu)
            d["imu_rate_hz"] = (len(self.imu) - 1) / (self.imu.t[-1] - self.imu.t[0])
            d["imu_t_range"] = [float(self.imu.t[0]), float(self.imu.t[-1])]
        return d


# ══════════════════════════════════════════════════════════════
#  读取
# ══════════════════════════════════════════════════════════════
_UNIT_TO_SEC = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
_GYRO_TO_RAD = {"rad/s": 1.0, "deg/s": np.pi / 180.0}
_ACCEL_TO_SI = {"m/s^2": 1.0, "g": 9.80665}


def load_imu_txt(path: str | Path, time_unit: str = "s",
                 gyro_unit: str = "rad/s", accel_unit: str = "m/s^2",
                 delimiter: str | None = None) -> ImuData:
    """读 ``timestamp gx gy gz ax ay az`` 的文本表（``#`` 开头为注释）。"""
    ts, gy, ac = [], [], []
    for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.replace(",", " ").split(delimiter) if delimiter else ln.split()
        parts = [p for p in parts if p != ""]
        if len(parts) < 7:
            continue
        try:
            v = [float(p) for p in parts[:7]]
        except ValueError:
            continue
        ts.append(v[0]); gy.append(v[1:4]); ac.append(v[4:7])
    if not ts:
        raise ValueError(f"{path} 里没解析到任何 IMU 行（期望 'timestamp gx gy gz ax ay az'）")
    return ImuData(
        t=np.asarray(ts, dtype=float) * _UNIT_TO_SEC[time_unit],
        gyro=np.asarray(gy, dtype=float) * _GYRO_TO_RAD[gyro_unit],
        accel=np.asarray(ac, dtype=float) * _ACCEL_TO_SI[accel_unit],
    )


def load_image_timestamps(path: str | Path, time_unit: str = "s") -> dict[str, float]:
    """读 ``<文件名> <时间戳>`` 的映射。"""
    out: dict[str, float] = {}
    for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 2:
            continue
        try:
            out[parts[0]] = float(parts[1]) * _UNIT_TO_SEC[time_unit]
        except ValueError:
            continue
    return out


def load_session(root: str | Path, image_dir: str = "images",
                 image_timestamps: str = "image_timestamps.txt",
                 imu_file: str = "imu.txt", time_unit: str = "s") -> Session:
    """按 README 约定的目录结构装载一次采集。"""
    root = Path(root).expanduser()
    imgs = sorted((root / image_dir).glob("*.png")) + sorted((root / image_dir).glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"{root/image_dir} 里没有图像")
    tsm = load_image_timestamps(root / image_timestamps, time_unit)
    ts = np.array([tsm[p.name] for p in imgs if p.name in tsm], dtype=float)
    if len(ts) != len(imgs):
        missing = [p.name for p in imgs if p.name not in tsm]
        raise ValueError(f"这些图像没有时间戳：{missing[:5]}（共 {len(missing)} 张）")
    imu = None
    p = root / imu_file
    if p.exists():
        imu = load_imu_txt(p, time_unit=time_unit)
    return Session(root=root, image_paths=imgs, image_ts=ts, imu=imu)


# ══════════════════════════════════════════════════════════════
#  结果写盘（YAML + JSON，且可被重新读回）
# ══════════════════════════════════════════════════════════════
def build_result(intrinsics: dict, extrinsic: dict | None, validation: dict | None,
                 meta: dict | None = None) -> dict:
    return {
        "meta": {"written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "convention_note": "T_IC 表示 相机系 C -> IMU 系 I 的变换",
                 **(meta or {})},
        "camera": intrinsics,
        "extrinsic": extrinsic or {},
        "validation": validation or {},
    }


def save_result(result: dict, out_dir: str | Path, yaml_name="calibration_result.yaml",
                json_name="calibration_result.json", keep_snapshots=True) -> list[Path]:
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, dumper in ((yaml_name, lambda d, f: yaml.safe_dump(
            d, f, allow_unicode=True, sort_keys=False, default_flow_style=None)),
            (json_name, lambda d, f: json.dump(d, f, ensure_ascii=False, indent=2))):
        p = out / name
        with p.open("w", encoding="utf-8") as f:
            dumper(result, f)
        written.append(p)
        if keep_snapshots:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            snap = out / f"{p.stem}_{stamp}{p.suffix}"
            snap.write_bytes(p.read_bytes())
            written.append(snap)
    return written


def load_result(path: str | Path) -> dict:
    """把结果读回来。YAML / JSON 都支持——任务书要求"结果能被程序重新读取"。"""
    p = Path(path).expanduser()
    txt = p.read_text(encoding="utf-8")
    return json.loads(txt) if p.suffix == ".json" else yaml.safe_load(txt)


def load_config(path: str | Path) -> dict:
    p = Path(path).expanduser()
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(p)
    cfg["_root"] = str(p.parent.parent)
    return cfg
