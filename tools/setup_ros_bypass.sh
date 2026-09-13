#!/usr/bin/env bash
# 免 sudo 准备 ROS 2 的 camera_calibration，用于"与成熟工具对比"这一项验证。
#
# 为什么需要这个脚本
# ------------------------------------------------------------------
# `ros-jazzy-camera-calibration` 需要 sudo 才能 apt install，而本机 sudo 要密码。
# 但 **camera_calibration 的核心（calibrator.py）是可以无头调用的** ——
# 它只依赖 cv2 / numpy / cv_bridge / rclpy，而这些本机全都装好了。
# 所以把 .deb 抽出来、加进 PYTHONPATH 就能用，不需要 root、不需要 ROS topic、
# 不需要 GUI、不需要相机。
#
# 副作用：只往 apt 缓存和 /tmp/roscc 里写东西，卸载就是 rm -rf /tmp/roscc。
#
# 用法：
#   bash tools/setup_ros_bypass.sh
#   # 然后（脚本会打印确切命令）：
#   .venv/bin/python tools/compare_with_ros.py --session data/synth_01 --pattern 9x6 --square-mm 25

set -euo pipefail

DEST="${ROS_CC_DEST:-/tmp/roscc}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== 1. 下载并解包 camera_calibration 到 $DEST ==="
rm -rf "$DEST"; mkdir -p "$DEST"; cd "$DEST"
apt-get download ros-jazzy-camera-calibration
dpkg -x ./*.deb ex
PKG_PATH="$DEST/ex/opt/ros/jazzy/lib/python3.12/site-packages"
[ -d "$PKG_PATH" ] || { echo "❌ 解包后找不到 python 包目录: $PKG_PATH" >&2; exit 1; }
echo "  ✅ $PKG_PATH"

echo
echo "=== 2. 检查依赖 ==="
missing=0
for p in ros-jazzy-cv-bridge ros-jazzy-image-transport ros-jazzy-message-filters \
         ros-jazzy-rclpy ros-jazzy-sensor-msgs; do
    if dpkg -l "$p" 2>/dev/null | grep -q '^ii'; then
        echo "  ✅ $p"
    else
        echo "  ❌ $p 未安装（需要 sudo apt install $p）"; missing=1
    fi
done
[ "$missing" -eq 0 ] || echo "  ⚠️  有依赖缺失，对照可能跑不起来"

echo
echo "=== 3. 补 Python 依赖 semver ==="
if "$ROOT/.venv/bin/python" -c "import semver" 2>/dev/null; then
    echo "  ✅ semver 已有"
else
    "$ROOT/.venv/bin/pip" install -q -i https://mirrors.ustc.edu.cn/pypi/simple semver
    echo "  ✅ semver 已装（USTC 镜像）"
fi

echo
echo "=== 4. 自检 ==="
PYTHONPATH="$PKG_PATH:/opt/ros/jazzy/lib/python3.12/site-packages:$ROOT/.venv/lib/python3.12/site-packages" \
  "$ROOT/.venv/bin/python" - <<'PY'
import sys
sys.path.insert(0, "/tmp/roscc/ex/opt/ros/jazzy/lib/python3.12/site-packages")
sys.path.insert(0, "/opt/ros/jazzy/lib/python3.12/site-packages")
try:
    from camera_calibration.calibrator import MonoCalibrator, ChessboardInfo
    print("  ✅ camera_calibration 可无头导入（不需要 ROS topic / GUI / 相机）")
except Exception as e:
    print(f"  ❌ 导入失败：{type(e).__name__}: {e}")
    raise SystemExit(1)
PY

cat <<EOF

=== 完成 ===

用法（不需要 source ROS，脚本内部会处理路径）：

  .venv/bin/python tools/compare_with_ros.py \\
      --session data/session_01 --pattern 9x6 --square-mm 25

它会：
  1. 用 ROS 自己的 collect_corners 检测角点（**独立检测，不是复用我们的**）；
  2. 跑 ROS 的标定；
  3. 与我们的 C1 结果做**投影函数级**对比（在视锥里撒点，比像素差），
     而不是只比 fx/fy 那几个数 —— 后者看不出畸变系数的相互抵消。

若换了机器或想重来：rm -rf $DEST 后重跑本脚本。
EOF
