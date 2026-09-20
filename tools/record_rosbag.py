#!/usr/bin/env python3
"""把一个采集 session 写成 **rosbag2**（标准 ROS 2 格式）。

为什么需要它（这是任务书"'与成熟工具对比'之外的第二条旁路"）
==================================================================
我们自己的标定链路是纯 Python 的，跑得又快又可控，但它**不是通用格式**：
别人拿到 `data/session_01/` 没法用 ROS 生态的工具复现。

写成 rosbag2 之后有三件事立刻成立：

1. **`ros2 bag play` + `camera_calibration` GUI 可以现场演示** ——
   评审时"用 ROS 官方工具再标一遍"是最直观的第三方对照；
2. **数据可被任何 ROS 工具消费**（`rqt_bag`、`ros2 topic echo`、
   `image_view`、RViz2），不需要我们解释自己的目录结构；
3. **时间戳语义变成标准语义**：`CameraInfo` 里可以带上我们标好的内参，于是
   "我们的结果" 与 "ROS 工具链" 之间有了一个标准接口。

时间戳约定（**这里有个坑，解释一下为什么不做"原样写"**）
------------------------------------------------------------------
标定真正用到的只有**相对**时间（τ 是一个相对偏置，外参只依赖相对运动），
而 `data/session_*/` 里的时间戳是 `time.monotonic()`（开机以来的秒数），
合成数据则从 0 附近开始。

**原样写进 bag 会坏掉**，实测：`starting_time.nanoseconds_since_epoch:
-637000000` —— session 从 0 附近开始时首条消息是**负数**，而 rosbag2 这个字段
不接受负值，读回时报的是
``Exception on parsing info file: yaml-cpp: error at line 7, column 30: bad conversion``，
完全看不出跟时间戳有关。所以必须整体做一次**常量平移**。

平移**不改变任何相对关系**，因此对 τ、外参、验证全部无损 ——
这是精确成立的性质，不是近似。默认平移到固定的 ``--start-epoch`` 锚点，
于是相同输入永远生成相同的 bag（可复现、可对比）。
绝对锚点只是装饰，**不代表真实采集时刻**。

用法::

    # 合成数据也能写（不需要任何硬件）—— 用来验证本工具本身
    .venv/bin/python tools/record_rosbag.py \\
        --session data/synth_user --out bags/synth_user

    # 用我们标好的内参填 CameraInfo（可选，但演示时很有用）
    .venv/bin/python tools/record_rosbag.py \\
        --session data/session_01 --out bags/session_01 \\
        --intrinsics results/calibration_result.yaml

    # 回放（另开终端）
    source /opt/ros/jazzy/setup.bash
    ros2 bag play bags/session_01
    ros2 run camera_calibration cameracalibrator \\
        --size 11x8 --square 0.020 image:=/cam/image_raw camera:=/cam
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calib"))

ROS_DISTRO_PKG = "/opt/ros/jazzy/lib/python3.12/site-packages"
ROS_DISTRO_LIB = "/opt/ros/jazzy/lib"
ROS_PREFIX = "/opt/ros/jazzy"

# 话题名。默认值刻意与 ROS 生态惯例一致，这样 camera_calibration 的
# 重映射参数最短：image:=/cam/image_raw camera:=/cam
TOPIC_IMAGE = "/cam/image_raw"
TOPIC_INFO = "/cam/camera_info"
TOPIC_IMU = "/imu/data"


def _ensure_ros_env() -> None:
    """让本进程能 import 并**真正用起来** rclpy / rosbag2_py。

    需要三样东西，缺一个都会以很奇怪的方式失败（都实测踩过）：

    1. ``LD_LIBRARY_PATH`` 含 ``/opt/ros/jazzy/lib`` —— ``rosbag2_py`` 的 C 扩展
       要在这里找 .so。它是动态链接器**启动时**读的，从 Python 里改没用，
       所以带对环境变量 **exec 重启自己一次**（与 compare_with_ros.py 同样的手法）。
    2. ``PYTHONPATH`` 含 distro 的 site-packages —— 否则 import 不到 rclpy。
    3. ``AMENT_PREFIX_PATH=/opt/ros/jazzy`` —— 这个最容易漏：
       ``rosbag2_py.SequentialWriter()`` 要靠它去查**存储插件**
       （``rosbag2_storage_default_plugins``），没设时报的是
       ``Unable to create class loader instance`` + 一个没有任何信息量的
       ``RuntimeError: std::exception``，看不出跟环境变量有关。
    """
    import os
    need_exec = ROS_DISTRO_LIB not in os.environ.get("LD_LIBRARY_PATH", "").split(":")
    need_exec = need_exec or not os.environ.get("AMENT_PREFIX_PATH")
    if need_exec:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = ROS_DISTRO_LIB + (
            ":" + os.environ["LD_LIBRARY_PATH"]
            if os.environ.get("LD_LIBRARY_PATH") else "")
        # AMENT_PREFIX_PATH 可能是多个前缀用 ':' 连接；把 jazzy 补进去而不是覆盖，
        # 以免破坏 overlay 工作区（~/ros2_ws）的既有设置。
        parts = [p for p in os.environ.get("AMENT_PREFIX_PATH", "").split(":") if p]
        if ROS_PREFIX not in parts:
            parts.append(ROS_PREFIX)
        env["AMENT_PREFIX_PATH"] = ":".join(parts)
        env.setdefault("ROS_DISTRO", "jazzy")
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    if ROS_DISTRO_PKG not in sys.path:
        sys.path.insert(0, ROS_DISTRO_PKG)


def _load_intrinsics(path: Path | None):
    """读我们自己的标定结果，取 (K, dist, size)；没有就返回 None。"""
    if path is None:
        return None
    from io_data import load_result

    d = load_result(path)
    c = d["camera"]
    i = c["intrinsics"]
    K = [float(i["fx"]), 0.0, float(i["cx"]),
         0.0, float(i["fy"]), float(i["cy"]),
         0.0, 0.0, 1.0]
    dist = [float(v) for v in c["distortion"]["coefficients"]]
    return K, dist, tuple(int(v) for v in c["image_size"])


def main() -> int:
    ap = argparse.ArgumentParser(
        description="把 session 写成 rosbag2（标准 ROS 2 格式）")
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", required=True, help="输出 bag 目录")
    ap.add_argument("--intrinsics", default=None,
                    help="我们的标定结果 yaml（可选，用于填 CameraInfo）")
    ap.add_argument("--frame-id", default="camera")
    ap.add_argument("--imu-frame-id", default="imu")
    ap.add_argument("--storage", default="sqlite3", choices=["sqlite3", "mcap"])
    ap.add_argument("--no-images", action="store_true",
                    help="只写 IMU/CameraInfo（图很大时用）")
    ap.add_argument("--max-imu", type=int, default=0,
                    help="只写前 N 条 IMU（0=全部）；调试用")
    # 固定锚点 → 相同输入产出相同 bag。选 2023-11-14 只是为了让 GUI 里
    # 显示的时间"看起来正常"；它**不代表真实采集时刻**（见文件头说明）。
    ap.add_argument("--start-epoch", type=float, default=1_700_000_000.0,
                    help="首条消息的绝对时间锚点（秒，Unix epoch）；"
                         "只影响显示，不影响任何相对时间")
    args = ap.parse_args()

    _ensure_ros_env()
    import cv2
    import rosbag2_py
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import CameraInfo, Image, Imu

    from io_data import load_session

    sess = load_session(args.session)
    n_img = len(sess)
    n_imu = len(sess.imu) if sess.imu is not None else 0
    print(f"数据: {sess.root}")
    print(f"  图像 {n_img} 帧，IMU {n_imu} 帧")

    # ── 常量时间平移：把所有时间戳挪到 start_epoch 之后 ────────────────────
    # 只做平移、不做缩放/重采样 → 任何**相对**时间（τ、帧间隔、IMU 步长）
    # 逐位不变。这是本工具对"不篡改数据"的保证的准确表述。
    cands = []
    if len(sess.image_ts):
        cands.append(float(np.min(sess.image_ts)))
    if n_imu:
        cands.append(float(np.min(sess.imu.t)))
    t_min = min(cands) if cands else 0.0
    shift = float(args.start_epoch) - t_min
    print(f"  原始首条时间戳 {t_min:.6f} s；平移 {shift:+.3f} s "
          f"→ 锚到 {args.start_epoch:.0f}（相对时间逐位不变）")

    def to_ns(t: float) -> int:
        return int(round((float(t) + shift) * 1e9))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        import shutil
        shutil.rmtree(out)

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(out), storage_id=args.storage),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"),
    )

    _topic_id = [0]

    def make_topic(name, typ):
        # TopicMetadata 只接受**位置参数**（id 在最前），用关键字会 TypeError。
        # id 由调用方分配，保证唯一即可。
        _topic_id[0] += 1
        return rosbag2_py.TopicMetadata(
            _topic_id[0], name, typ, "cdr", [])

    if not args.no_images:
        writer.create_topic(make_topic(TOPIC_IMAGE, "sensor_msgs/msg/Image"))
    writer.create_topic(make_topic(TOPIC_INFO, "sensor_msgs/msg/CameraInfo"))
    if n_imu:
        writer.create_topic(make_topic(TOPIC_IMU, "sensor_msgs/msg/Imu"))

    intr = _load_intrinsics(Path(args.intrinsics)) if args.intrinsics else None
    if intr is None and args.intrinsics:
        print(f"  ⚠️ 读不到内参 {args.intrinsics}，CameraInfo 将留空")
    elif intr is not None:
        print(f"  ✅ CameraInfo 带上我们的内参 K={intr[0][0]:.2f}...")

    # ── 写图像 + CameraInfo ────────────────────────────────────────────
    n_written_img = 0
    size = intr[2] if intr else None
    if not args.no_images:
        for p, t in zip(sess.image_paths, sess.image_ts):
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            h, w = img.shape[:2]
            t_ns = to_ns(t)

            msg = Image()
            msg.header.stamp.sec = t_ns // 1_000_000_000
            msg.header.stamp.nanosec = t_ns % 1_000_000_000
            msg.header.frame_id = args.frame_id
            msg.height, msg.width = int(h), int(w)
            # Mono8 = 1 字节/像素；本项目工业相机固定 Mono8
            msg.encoding = "mono8" if img.ndim == 2 else "bgr8"
            msg.is_bigendian = 0
            msg.step = int(w * (1 if img.ndim == 2 else 3))
            msg.data = np.ascontiguousarray(img).tobytes()
            writer.write(TOPIC_IMAGE, serialize_message(msg), t_ns)

            ci = CameraInfo()
            ci.header = msg.header
            ci.height, ci.width = int(h), int(w)
            if intr:
                K, dist, _ = intr
                ci.k = [float(v) for v in K]
                ci.d = [float(v) for v in dist]
                ci.distortion_model = "plumb_bob"
                fx, fy, cx, cy = K[0], K[4], K[2], K[5]
                ci.p = [fx, 0.0, cx, 0.0,
                        0.0, fy, cy, 0.0,
                        0.0, 0.0, 1.0, 0.0]
                ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            writer.write(TOPIC_INFO, serialize_message(ci), t_ns)
            n_written_img += 1
            if size is None:
                size = (w, h)

    # ── 写 IMU ────────────────────────────────────────────────────────
    n_written_imu = 0
    if n_imu:
        imu = sess.imu
        m = n_imu if args.max_imu <= 0 else min(args.max_imu, n_imu)
        for k in range(m):
            t_ns = to_ns(imu.t[k])
            msg = Imu()
            msg.header.stamp.sec = t_ns // 1_000_000_000
            msg.header.stamp.nanosec = t_ns % 1_000_000_000
            msg.header.frame_id = args.imu_frame_id
            # H7 的坐标系就是 msg 要求的右手系，直接填；单位已是 SI。
            msg.angular_velocity.x = float(imu.gyro[k, 0])
            msg.angular_velocity.y = float(imu.gyro[k, 1])
            msg.angular_velocity.z = float(imu.gyro[k, 2])
            msg.linear_acceleration.x = float(imu.accel[k, 0])
            msg.linear_acceleration.y = float(imu.accel[k, 1])
            msg.linear_acceleration.z = float(imu.accel[k, 2])
            # 这块板的磁力计恒为 0、yaw 不可观测，所以**不填** orientation
            # （填了就相当于声称有姿态观测，会让下游误用）。
            # 协方差第 0 项 = -1 表示 "该字段无数据"，这是 ROS 的标准约定。
            msg.orientation_covariance[0] = -1.0
            writer.write(TOPIC_IMU, serialize_message(msg), t_ns)
            n_written_imu += 1

    del writer

    # ── 校验：把 bag 重新读回来，确认真的写进去了 ─────────────────────
    print(f"\n已写入 {out}")
    print(f"  图像 {n_written_img}   CameraInfo {n_written_img}   IMU {n_written_imu}")
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(out), storage_id=args.storage),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"),
    )
    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    counts: dict[str, int] = {k: 0 for k in topics}
    t_first: dict[str, int] = {}
    t_last: dict[str, int] = {}
    while reader.has_next():
        name, _data, t_ns = reader.read_next()
        counts[name] = counts.get(name, 0) + 1
        t_first.setdefault(name, t_ns)
        t_last[name] = t_ns
    print("\n回读校验：")
    ok = True
    for name, typ in sorted(topics.items()):
        c = counts.get(name, 0)
        span = (t_last.get(name, 0) - t_first.get(name, 0)) / 1e9
        print(f"  {name:<20} {typ:<28} {c:>6} 条   跨度 {span:8.3f} s")
        if c == 0:
            ok = False
    if n_written_img and counts.get(TOPIC_IMAGE, 0) != n_written_img:
        print("  ❌ 图像条数与写入不符"); ok = False
    if n_written_imu and counts.get(TOPIC_IMU, 0) != n_written_imu:
        print("  ❌ IMU 条数与写入不符"); ok = False
    print("  ✅ bag 可被 rosbag2 重新读回，条数一致" if ok else "  ❌ 校验失败")

    print("\n回放与演示：")
    print("  source /opt/ros/jazzy/setup.bash")
    print(f"  ros2 bag play {out}")
    print("  ros2 run camera_calibration cameracalibrator \\")
    print("      --size <内角点如 11x8> --square 0.020 "
          "image:=/cam/image_raw camera:=/cam")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
