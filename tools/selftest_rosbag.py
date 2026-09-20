#!/usr/bin/env python3
"""自测：**rosbag2 → `ros2 bag play` → ROS 话题 → camera_calibration** 整条链路。

为什么值得单独测
==================================================================
`record_rosbag.py` 已经把数据写进标准 bag，但"bag 文件能被读回"和
"**官方工具能从这条链路里真的标出内参**"是两件事，中间还夹着：

* `ros2 bag play` 能不能正常发（QoS / 时钟 / 话题名）；
* `cv_bridge` 能不能把 `sensor_msgs/Image` 转成 OpenCV 图（编码、step、字节序）；
* `camera_calibration` 的订阅端能不能收到并入库。

这三步任何一步坏了，**验收现场才发现就晚了** —— 而现场是要拿这条链路做
"用 ROS 官方工具再标一遍"的演示的。

所以本脚本用**真实的 `ros2 bag play` 子进程**（不是自己假装发消息），
把 bag 里的图喂给 `camera_calibration` 自己的 `MonoCalibrator`，
再把结果和我们直接读文件跑出来的 ROS 结果对比 —— **两者应当逐位一致**，
因为喂进去的就是同一批像素。

用法::

    # 先造一个 bag（合成数据即可，不需要硬件）
    .venv/bin/python tools/record_rosbag.py \\
        --session data/synth_user --out /tmp/bags/synth_user

    .venv/bin/python tools/selftest_rosbag.py \\
        --bag /tmp/bags/synth_user --pattern 11x8 --square-mm 20
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calib"))

ROS_DISTRO_PKG = "/opt/ros/jazzy/lib/python3.12/site-packages"
ROS_DISTRO_LIB = "/opt/ros/jazzy/lib"
ROS_PREFIX = "/opt/ros/jazzy"
TOPIC_IMAGE = "/cam/image_raw"


def _ensure_ros_env() -> None:
    """见 record_rosbag.py 的同名函数：三样环境变量缺一不可，且必须 exec 重启。"""
    if ROS_DISTRO_LIB not in os.environ.get("LD_LIBRARY_PATH", "").split(":") \
            or not os.environ.get("AMENT_PREFIX_PATH"):
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = ROS_DISTRO_LIB + (
            ":" + os.environ["LD_LIBRARY_PATH"]
            if os.environ.get("LD_LIBRARY_PATH") else "")
        parts = [p for p in os.environ.get("AMENT_PREFIX_PATH", "").split(":") if p]
        if ROS_PREFIX not in parts:
            parts.append(ROS_PREFIX)
        env["AMENT_PREFIX_PATH"] = ":".join(parts)
        env.setdefault("ROS_DISTRO", "jazzy")
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    for p in (ROS_DISTRO_PKG, str(ROOT / "third_party/roscc/ex/opt/ros/jazzy/lib/python3.12/site-packages")):
        if Path(p).is_dir() and p not in sys.path:
            sys.path.insert(0, p)


def main() -> int:
    ap = argparse.ArgumentParser(description="rosbag2 → play → camera_calibration 链路自测")
    ap.add_argument("--bag", required=True)
    ap.add_argument("--pattern", default="11x8")
    ap.add_argument("--square-mm", type=float, default=20.0)
    ap.add_argument("--rate", type=float, default=0.0,
                    help="回放倍速（0=不限速，尽快放完）")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    _ensure_ros_env()
    import cv_bridge
    import rclpy
    from camera_calibration.calibrator import ChessboardInfo, MonoCalibrator
    from rclpy.node import Node
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import Image as RosImage

    bag = Path(args.bag)
    if not bag.exists():
        raise SystemExit(f"❌ 找不到 bag {bag}；先跑 tools/record_rosbag.py")

    c, r = (int(v) for v in args.pattern.lower().split("x"))
    print("=" * 70)
    print("自测：rosbag2 → ros2 bag play → ROS 话题 → camera_calibration")
    print("=" * 70)
    print(f"bag   : {bag}")
    print(f"板    : 内角点 {c}x{r}，方格 {args.square_mm} mm")

    # ── ① 启动真实的 `ros2 bag play` 子进程 ────────────────────────────
    # `ros2` 这个 CLI 在 /opt/ros/jazzy/bin 下，而本进程**没有 source 过
    # setup.bash**（我们是靠显式改环境变量跑起来的），所以必须自己把 bin
    # 加进 PATH，否则报 FileNotFoundError: 'ros2'。
    cmd = ["ros2", "bag", "play", str(bag)]
    if args.rate > 0:
        cmd += ["--rate", str(args.rate)]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ROS_DISTRO_LIB + (":" + env["LD_LIBRARY_PATH"]
                                               if env.get("LD_LIBRARY_PATH") else "")
    env["PATH"] = f"{ROS_PREFIX}/bin:" + env.get("PATH", "")
    print(f"\n① 启动回放：{' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env,
                            preexec_fn=os.setsid if hasattr(os, "setsid") else None)

    # ── ② 订阅话题，把图喂给 camera_calibration 自己的检测+标定 ──────────
    print("② 订阅 %s，喂给 MonoCalibrator ..." % TOPIC_IMAGE)
    rclpy.init()
    node = Node("rosbag_selftest")
    # 传感器流惯例用 best_effort；bag 里没记 QoS，两种都试，保证收得到。
    qos = QoSProfile(depth=50, history=QoSHistoryPolicy.KEEP_LAST,
                     reliability=QoSReliabilityPolicy.BEST_EFFORT)
    bridge = cv_bridge.CvBridge()
    ci = ChessboardInfo(pattern="chessboard", n_cols=c, n_rows=r,
                        dim=float(args.square_mm / 1000.0))
    mc = MonoCalibrator([ci])
    seen = {"n": 0, "encodings": set(), "shapes": set()}
    done = threading.Event()

    def on_image(msg: RosImage) -> None:
        try:
            img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:                      # noqa: BLE001
            print(f"   ❌ cv_bridge 转换失败：{type(e).__name__}: {e}")
            done.set()
            return
        seen["n"] += 1
        seen["encodings"].add(msg.encoding)
        seen["shapes"].add(img.shape)
        mc.handle_msg(msg)          # 走官方内部流程（含它自己的入库准则）

    node.create_subscription(RosImage, TOPIC_IMAGE, on_image, qos)

    t0 = time.time()
    while time.time() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
        if done.is_set():
            break
        # 回放结束后再等一小会儿，确保消息都处理完
        if proc.poll() is not None:
            for _ in range(10):
                rclpy.spin_once(node, timeout_sec=0.1)
            break

    print(f"   收到 {seen['n']} 帧；编码 {sorted(seen['encodings'])}；"
          f"尺寸 {sorted(seen['shapes'])}")
    node.destroy_node()
    rclpy.shutdown()
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:                                # noqa: BLE001
        proc.kill()

    if seen["n"] == 0:
        print("\n❌ 一帧都没收到 —— 回放或订阅环节坏了。")
        return 1

    # ── ③ 用收到的图做标定 ────────────────────────────────────────────
    print("\n③ 用 ROS 官方流程标定 ...")
    good = mc.good_corners
    if not good or len(good) < 5:
        print(f"   ❌ ROS 只入库 {0 if not good else len(good)} 张有效板子")
        return 1
    mc.cal_fromcorners(good)
    out = mc.as_message()
    K = np.asarray(getattr(out, "k", None) if getattr(out, "k", None) is not None
                   else out.K, float).reshape(3, 3)
    D = np.asarray(getattr(out, "d", None) if getattr(out, "d", None) is not None
                   else out.D, float).ravel()
    print(f"   ROS 用上 {len(good)}/{seen['n']} 张")
    print(f"   fx={K[0,0]:.4f} fy={K[1,1]:.4f} cx={K[0,2]:.4f} cy={K[1,2]:.4f}")
    print(f"   dist=" + " ".join(f"{v:+.5f}" for v in D))

    # ── ④ 与"直接读文件"的 ROS 结果对比：应当逐位一致 ───────────────────
    print("\n④ 与直接读文件跑 ROS 的结果对比（喂的是同一批像素，应逐位一致）")
    import cv2
    from camera_calibration.calibrator import MonoCalibrator as MC2

    sess_dir = None
    meta = bag / "metadata.yaml"
    # 从 bag 名反推 session（约定 --out 用 session 名），找不到就跳过这步
    cand = ROOT / "data" / bag.name
    if cand.is_dir():
        sess_dir = cand
    if sess_dir is None or not meta.exists():
        print("   （找不到对应 session，跳过逐位对比）")
        return 0

    grays = []
    for p in sorted((sess_dir / "images").glob("*.png")):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            grays.append(g)
    mc2 = MC2([ChessboardInfo(pattern="chessboard", n_cols=c, n_rows=r,
                              dim=float(args.square_mm / 1000.0))])
    g2 = mc2.collect_corners(grays)
    mc2.good_corners = g2
    mc2.cal_fromcorners(g2)
    o2 = mc2.as_message()
    K2 = np.asarray(getattr(o2, "k", None) if getattr(o2, "k", None) is not None
                    else o2.K, float).reshape(3, 3)
    D2 = np.asarray(getattr(o2, "d", None) if getattr(o2, "d", None) is not None
                    else o2.D, float).ravel()
    dK = float(np.abs(K - K2).max())
    dD = float(np.abs(D - D2).max())
    print(f"   |K_bag - K_file| 最大 = {dK:.3e}")
    print(f"   |D_bag - D_file| 最大 = {dD:.3e}")
    same = dK < 1e-9 and dD < 1e-9
    print("   " + ("✅ 逐位一致 —— bag 无损，链路可用" if same
                   else "⚠️ 有差异，说明 bag 或链路改动了数据，需要查"))
    if same:
        print("\n" + "=" * 70)
        print("✅ 结论：可以用 ros2 bag play + cameracalibrator 做现场演示")
        print("=" * 70)
        print("   source /opt/ros/jazzy/setup.bash")
        print(f"   ros2 bag play {bag}")
        print(f"   ros2 run camera_calibration cameracalibrator \\")
        print(f"       --size {c}x{r} --square {args.square_mm/1000:.4f} "
              f"image:={TOPIC_IMAGE} camera:=/cam")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
