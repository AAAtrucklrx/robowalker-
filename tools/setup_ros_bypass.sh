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
# 为什么装在项目里而不是 /tmp（这里踩过坑）
# ------------------------------------------------------------------
# 最初 DEST=/tmp/roscc。结果 /tmp 被系统清理后，`compare_with_ros.py` 直接
# 报"用不了 camera_calibration"——**而它是"与成熟工具对比"这项验证的唯一入口**，
# 静默失效等于验收时少一项。
# 现在改成两条保险：
#   1. DEST 默认落在项目内的 third_party/roscc，跟项目一起备份；
#   2. 下载的 .deb **缓存**在 third_party/roscc/cache/，重装/换机时断网也能重建。
# 卸载就是 rm -rf third_party/roscc。
#
# 用法：
#   bash tools/setup_ros_bypass.sh
#   # 然后（脚本会打印确切命令）：
#   .venv/bin/python tools/compare_with_ros.py --session data/synth_01 --pattern 9x6 --square-mm 25

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROS_CC_DEST:-$ROOT/third_party/roscc}"
CACHE="$DEST/cache"
PKG_REL="ex/opt/ros/jazzy/lib/python3.12/site-packages"

echo "=== 1. 准备 camera_calibration 到 $DEST ==="
mkdir -p "$CACHE"

# 已有缓存的 .deb 就直接复用（断网可用），否则才去源上拉。
if compgen -G "$CACHE/*.deb" > /dev/null; then
    echo "  ✅ 复用缓存：$(basename "$(ls "$CACHE"/*.deb | head -1)")"
else
    echo "  ⬇️  缓存为空，从源下载（仅此一次，之后断网也能重建）"
    ( cd "$CACHE" && apt-get download ros-jazzy-camera-calibration )
fi

# 只重建 ex/，**不动 cache/**，否则每次都要重新下载。
rm -rf "$DEST/ex"
( cd "$DEST" && dpkg -x "$CACHE"/*.deb ex )
PKG_PATH="$DEST/$PKG_REL"
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
# 这里必须用 $PKG_PATH，不能写死路径 —— 之前写死了 /tmp/roscc，
# 于是改了 DEST 之后自检仍然去看旧路径，测的不是真正会用到的那份。
PYTHONPATH="$PKG_PATH:/opt/ros/jazzy/lib/python3.12/site-packages" \
LD_LIBRARY_PATH="/opt/ros/jazzy/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$ROOT/.venv/bin/python" - <<PY
import sys
sys.path.insert(0, r"$PKG_PATH")
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
      --session data/session_01 --pattern 11x8 --square-mm 20

它会：
  1. 用 ROS 自己的 collect_corners 检测角点（**独立检测，不是复用我们的**）；
  2. 跑 ROS 的标定；
  3. 与我们的 C1 结果做**投影函数级**对比（在视锥里撒点，比像素差），
     而不是只比 fx/fy 那几个数 —— 后者看不出畸变系数的相互抵消。

若想重来：rm -rf "$DEST" 后重跑本脚本（缓存没了才会再联网）。
EOF
