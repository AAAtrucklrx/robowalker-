#!/usr/bin/env bash
# 一次性修好串口与 USB 设备的访问权限，供 RoboWalker 标定/培训任务长期使用。
#
# 解决的问题：
#   1. /dev/ttyACM0（STM32 虚拟串口 IMU）是 root:dialout 660，普通用户读不了
#   2. 每次插拔后设备名会在 ttyACM0/ttyACM1 之间漂移
#   3. 后续还会插更多设备（IMU、控制板、USB 转串口），不想每次都 chmod
#
# 用法（需要 sudo 密码，请在你自己的终端里执行）：
#   sudo bash ~/calib_ws/tools/setup_permissions.sh
#
# 执行后按提示重新插拔一次设备即可，无需重新登录。

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "这个脚本需要 root 权限运行：" >&2
    echo "  sudo bash $0" >&2
    exit 1
fi

TARGET_USER="${SUDO_USER:-${1:-}}"
if [[ -z "$TARGET_USER" || "$TARGET_USER" == "root" ]]; then
    echo "无法确定要授权给哪个用户。用法: sudo bash $0 <用户名>" >&2
    exit 1
fi

echo "=== 授权用户: $TARGET_USER ==="

# ── 1. 把用户加入 dialout（串口）与 plugdev（USB 设备）组 ──────────────
for grp in dialout plugdev video; do
    if getent group "$grp" >/dev/null; then
        if id -nG "$TARGET_USER" | tr ' ' '\n' | grep -qx "$grp"; then
            echo "  [$grp] 已在组内，跳过"
        else
            usermod -aG "$grp" "$TARGET_USER"
            echo "  [$grp] 已加入"
        fi
    fi
done

# ── 2. 写 udev 规则：一次性覆盖后续所有同类设备 ────────────────────
RULE=/etc/udev/rules.d/99-robowalker-devices.rules
cat > "$RULE" <<EOF
# RoboWalker 算法组标定/训练设备权限规则
# 由 ~/calib_ws/tools/setup_permissions.sh 生成，可安全删除。

# STM32 虚拟串口（IMU 板 / 控制板）
#   0483:5740 = 通用 STM32 Virtual COMPort
#   0483:6666 = H7_IMU_With_EKF —— 当前在用的这块 IMU 板（2026-09-13 实测）
SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", MODE="0666", GROUP="dialout", SYMLINK+="rw_imu"
SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="6666", MODE="0666", GROUP="dialout", SYMLINK+="rw_imu"

# 通用 USB 转串口芯片：CH340 / CH341 / CP210x / FTDI / Prolific
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", MODE="0666", GROUP="dialout", SYMLINK+="rw_serial_ch340"
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", MODE="0666", GROUP="dialout", SYMLINK+="rw_serial_cp210x"
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", MODE="0666", GROUP="dialout", SYMLINK+="rw_serial_ftdi"
SUBSYSTEM=="tty", ATTRS{idVendor}=="067b", MODE="0666", GROUP="dialout", SYMLINK+="rw_serial_pl2303"

# 兜底：任何 USB 串口设备都给读写权限（本地开发机，风险可接受；如需收紧请删掉这行）
SUBSYSTEM=="tty", KERNEL=="ttyUSB*", MODE="0666", GROUP="dialout"
SUBSYSTEM=="tty", KERNEL=="ttyACM*", MODE="0666", GROUP="dialout"

# 海康机器人 U3V 工业相机（MV-CS020-10UC 等），VID = 2bdf
SUBSYSTEM=="usb", ATTRS{idVendor}=="2bdf", MODE="0666", GROUP="plugdev"
EOF
echo "  已写入 udev 规则: $RULE"

# ── 3. 立即生效 ─────────────────────────────────────────────────
udevadm control --reload-rules
udevadm trigger --subsystem-match=tty
udevadm trigger --subsystem-match=usb
echo "  udev 规则已重新加载"

# 对已经插入的设备立即补权限，省去拔插
sleep 1
for dev in /dev/ttyACM* /dev/ttyUSB*; do
    [[ -e "$dev" ]] && chmod 0666 "$dev" && echo "  已就地放开权限: $dev"
done

# ── 4. 结果核对 ─────────────────────────────────────────────────
echo
echo "=== 核对结果 ==="
id -nG "$TARGET_USER" | tr ' ' '\n' | grep -E "dialout|plugdev|video" | sed 's/^/  组: /'
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | sed 's/^/  /' || echo "  （当前没有串口设备）"
ls -l /dev/rw_imu 2>/dev/null | sed 's/^/  稳定别名: /' || true

cat <<'EOF'

=== 下一步 ===
1. 如果刚才没插设备，现在插上；已插的可以拔掉重插一次（让 udev 规则套用）。
2. 回到终端验证：
     ls -l /dev/rw_imu /dev/ttyACM0   # 两个都应该在，且是 crw-rw-rw-
     python3 ~/calib_ws/calib/imu_h7.py --port /dev/rw_imu --seconds 3

   ⚠️ 不要用 `cat /dev/ttyACM0` 验证。tty 行规程会篡改二进制流，看起来
   像"帧长忽长忽短"的变长协议，能把人带进沟里。要用 pyserial，或先执行
   `stty -F /dev/ttyACM0 raw` 再读。
3. dialout 组的新身份要重新登录才完全生效；udev 规则是立即生效的，
   所以本次不重新登录也能用（只要规则里的 MODE=0666 生效）。
EOF
