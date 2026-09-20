#!/usr/bin/env python3
"""把 EuRoC MAV 数据集（ROS1 bag）转成本项目管线的 session 格式。

为什么做这个
==================================================================
我们缺**真实的相机-IMU 采集数据**（受光照、支架、人手限制）。EuRoC 是
视觉惯性里程计领域最经典的公开数据集之一，而且：

* **图像 + IMU 同步、时间戳可信**；
* 官方**公布了标定结果**（用 Leica 跟踪仪 + Kalibr 做的），可以拿来做对照；
* 用 **AprilGrid** 靶标（6x6 个 AprilTag 36h11），靶标朝下也能检。

所以它能补上我们唯一缺的那一环：**在真实相机-IMU 数据上验证 C1/C2/C3**。

数据怎么来的
------------------------------------------------------------------
ETH 原始主机（robotics.ethz.ch）在国内连不上。实际用的是 HuggingFace 上的
镜像 `kavehsgh/EuRoC_MAV_Dataset_Machine_Hall_Easy_01`，而 HF 直连只有
~64 KB/s，必须走中文镜像 **hf-mirror.com**（实测 2.9 MB/s，2.67 GB 约 15 分钟）。

    curl -L -C - -o MH_01_easy.bag \\
      https://hf-mirror.com/datasets/kavehsgh/EuRoC_MAV_Dataset_Machine_Hall_Easy_01/resolve/main/MH_01_easy.bag

用法::

    # 先看 bag 里有什么
    .venv/bin/python tools/euroc_to_session.py --bag data/euroc/MH_01_easy.bag --list

    # 转换（只取前 N 秒可加快）
    .venv/bin/python tools/euroc_to_session.py --bag data/euroc/MH_01_easy.bag \\
        --out data/euroc_mh01 --cam /cam0/image_raw --imu /imu0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def stamp_of(msg) -> float:
    """取**消息头时间戳**（秒）。

    ⚠️ 这是本工具最重要的一个细节。EuRoC 的 bag 里有两套时间：
      · bag 录制时间（``reader.messages()`` 给的 ts）
      · 消息头 ``msg.header.stamp`` ← **这个才是真实采样时刻**

    实测（/imu0 前 400 条）：
        bag 录制时间   步长中位  86.3 µs，标准差 **±14647 µs**  ← 完全混乱
        header 时间戳  步长中位 **5000.1 µs**，标准差 **±0.1 µs** ← 精确 200 Hz

    用 bag 时间的后果：C2 时间对齐彻底失效、C3 全废，**而且不会报错** ——
    数据看着"有"，时间轴却是错的。所以这里必须用 header。
    """
    h = msg.header.stamp
    return float(h.sec) + float(h.nanosec) * 1e-9


def open_bag(bag: Path):
    from rosbags.highlevel import AnyReader

    return AnyReader([bag])


def list_topics(bag: Path) -> None:
    with open_bag(bag) as reader:
        print(f"bag: {bag}")
        print(f"时长: {(reader.end_time - reader.start_time)/1e9:.1f} s")
        print(f"\n{'话题':<28}{'类型':<38}{'条数':>10}  频率")
        for c in reader.connections:
            n = c.msgcount
            span = max((reader.end_time - reader.start_time) / 1e9, 1e-9)
            print(f"  {c.topic:<26}{c.msgtype:<38}{n:>10}  {n/span:7.1f} Hz")


def convert(bag: Path, out: Path, cam_topic: str, imu_topic: str,
            max_seconds: float = 0.0, max_images: int = 0,
            cam_index: int = 0) -> int:
    """把 bag 转成 session 目录。

    ``cam_index``：EuRoC 的 cam0/cam1 都是 image_raw，若话题名相同（不同 bag），
    用序号区分；默认 0。
    """
    import cv2

    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)

    with open_bag(bag) as reader:
        t0 = reader.start_time
        t_end = (t0 + int(max_seconds * 1e9)) if max_seconds > 0 else reader.end_time
        print(f"转换 {'全部' if max_seconds <= 0 else f'前 {max_seconds:.0f} s'}")
        print(f"  相机话题 {cam_topic}")
        print(f"  IMU 话题 {imu_topic}")

        # ── 图像 ────────────────────────────────────────────
        img_rows: list[tuple[str, float]] = []
        k = 0
        for conn, ts, raw in reader.messages(
                connections=[c for c in reader.connections
                             if c.topic == cam_topic]):
            if ts > t_end:
                break
            msg = reader.deserialize(raw, conn.msgtype)
            h, w = int(msg.height), int(msg.width)
            enc = str(msg.encoding).lower()
            buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            if enc in ("mono8", "8uc1"):
                img = buf[: h * w].reshape(h, w)
            elif enc in ("bayer_rggb8", "bayer_bggr8", "bayer_gbrg8",
                         "bayer_grbg8"):
                # EuRoC 的 cam0/cam1 是 BayerRG8，转灰度（不做去马赛克，
                # 因为标定只需要灰度角点，去马赛克反而引入插值模糊）
                img = buf[: h * w].reshape(h, w)
            elif enc in ("rgb8", "bgr8"):
                img = buf[: h * w * 3].reshape(h, w, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if enc == "rgb8"
                                   else cv2.COLOR_BGR2GRAY)
            else:
                img = buf[: h * w].reshape(h, w)
            name = f"{k:06d}.png"
            cv2.imwrite(str(out / "images" / name), img)
            img_rows.append((name, stamp_of(msg)))
            k += 1
            if max_images and k >= max_images:
                break
            if k % 200 == 0:
                print(f"    图像 {k} ...", flush=True)

        print(f"  图像共 {k} 张")

        # ── IMU ─────────────────────────────────────────────
        imu_rows = []
        for conn, ts, raw in reader.messages(
                connections=[c for c in reader.connections
                             if c.topic == imu_topic]):
            if ts > t_end:
                break
            msg = reader.deserialize(raw, conn.msgtype)
            g = msg.angular_velocity
            a = msg.linear_acceleration
            imu_rows.append((stamp_of(msg), g.x, g.y, g.z, a.x, a.y, a.z))
        print(f"  IMU 共 {len(imu_rows)} 条")

    if not img_rows:
        print(f"❌ 没读到图像。确认话题名（用 --list 看）"); return 2
    if not imu_rows:
        print(f"❌ 没读到 IMU。确认话题名（用 --list 看）"); return 2

    # ── 统一时钟 ────────────────────────────────────────────
    # EuRoC 的时间戳是「Unix epoch 纳秒」。我们的管线要求**单调时钟**语义
    # （见 io_data 的说明），但这里只要**相对时间正确**就行 —— 标定用的
    # τ 是相对偏置、外参只依赖相对运动。所以整体平移到一个正的基准时刻，
    # 相对关系逐位不变。
    t_min = min(img_rows[0][1], imu_rows[0][0])
    with (out / "image_timestamps.txt").open("w") as f:
        f.write("# filename timestamp_seconds (monotonic clock, unit: s)\n")
        for name, t in img_rows:
            f.write(f"{name} {t - t_min:.9f}\n")

    with (out / "imu.txt").open("w") as f:
        f.write("# timestamp gx gy gz ax ay az   "
                "(s, rad/s x3, m/s^2 x3)\n")
        for r in imu_rows:
            f.write(f"{r[0]-t_min:.9f} " + " ".join(f"{v:.9g}" for v in r[1:])
                    + "\n")

    d_img = np.diff([t for _, t in img_rows])
    d_imu = np.diff([r[0] for r in imu_rows])
    (out / "README.md").write_text(
        f"# EuRoC → session（{bag.name}）\n\n"
        f"- 来源：`{bag}`\n"
        f"- 相机话题：`{cam_topic}`\n"
        f"- IMU 话题：`{imu_topic}`\n"
        f"- 图像 {len(img_rows)} 张，帧间隔中位 {np.median(d_img)*1000:.2f} ms"
        f"（{1/np.median(d_img):.1f} Hz）\n"
        f"- IMU {len(imu_rows)} 条，步长中位 {np.median(d_imu)*1e6:.1f} µs"
        f"（{1/np.median(d_imu):.1f} Hz）\n"
        f"- 时间戳已整体平移到 0 起（相对关系不变）\n\n"
        f"⚠️ 靶标是 **AprilGrid**（6x6 AprilTag 36h11），不是棋盘格。\n"
    )
    # ── 自动检查：IMU 步长必须均匀 ────────────────────────────
    # 用 bag 录制时间会导致步长混乱（实测标准差 ±14.6 ms）。这类错误
    # **不会报错、数据看着也有**，但时间轴是错的 → 必须自动拦住。
    rel = np.median(d_imu)
    jitter = float(np.std(d_imu) / max(rel, 1e-12))
    print(f"\n时间戳检查：")
    print(f"  IMU 步长中位 {rel*1e6:.1f} µs，相对抖动 {jitter*100:.2f}%")
    if jitter > 0.05:
        print(f"  ⚠️  抖动过大！可能用了 bag 录制时间而不是 header 时间戳")
    else:
        print(f"  ✅ 步长均匀（{1/rel:.1f} Hz）")

    print(f"\n✅ 已写入 {out}")
    print(f"   图像 {len(img_rows)} 张，帧间隔 {np.median(d_img)*1000:.2f} ms")
    print(f"   IMU {len(imu_rows)} 条，步长 {np.median(d_imu)*1e6:.1f} µs")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="EuRoC bag → 本项目 session")
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cam", default="/cam0/image_raw")
    ap.add_argument("--imu", default="/imu0")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="只要前 N 秒（0=全部）")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="只列出话题")
    args = ap.parse_args()

    bag = Path(args.bag).expanduser()
    if not bag.exists():
        print(f"❌ 找不到 {bag}")
        print("   下载：curl -L -C - -o data/euroc/MH_01_easy.bag \\")
        print("     https://hf-mirror.com/datasets/kavehsgh/"
              "EuRoC_MAV_Dataset_Machine_Hall_Easy_01/resolve/main/MH_01_easy.bag")
        return 2

    if args.list:
        list_topics(bag)
        return 0

    out = Path(args.out) if args.out else ROOT / "data" / (bag.stem + "_session")
    return convert(bag, out, args.cam, args.imu,
                   args.max_seconds, args.max_images)


if __name__ == "__main__":
    raise SystemExit(main())
