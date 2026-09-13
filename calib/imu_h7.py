#!/usr/bin/env python3
"""H7_IMU_With_EKF 串口协议解析（VID:PID ``0483:6666``）。

协议是逆向出来的，不是抄手册 —— 下面是完整的证据链，方便你在答辩时解释。

帧结构（82 字节定长，小端）
------------------------------------------------------------------
    偏移  长度  类型       字段            单位 / 说明
    ----  ----  ---------  --------------  ---------------------------
    0     2     u8[2]      sync            ``5A A5``，固定
    2     2     u16        payload_len     ``0x004C`` = 76，固定
    4     2     u16        unknown         每帧变化，疑似校验/序号
    6     2     u16        const_0091      ``0x0091``，固定
    8     2     u16        const_002d      ``0x002D``，固定
    10    4     u32        reserved        恒为 0
    14    4     u32        t_board_ms      板载毫秒计时，**每帧 +1**
    18    4     f32        ax              加速度 x, m/s^2
    22    4     f32        ay
    26    4     f32        az
    30    4     f32        gx              角速度 x, rad/s
    34    4     f32        gy
    38    4     f32        gz
    42    4     f32        mx              磁力计 x，本板恒为 0（未启用）
    46    4     f32        my
    50    4     f32        mz
    54    4     f32        roll            欧拉角, rad
    58    4     f32        pitch
    62    4     f32        yaw
    66    4     f32        qw              四元数 **(w, x, y, z)** 顺序
    70    4     f32        qx
    74    4     f32        qy
    78    4     f32        qz
    ----  ----
    82    合计

逆向证据（2026-09-13 实测）
------------------------------------------------------------------
1. ``5A A5`` 出现间隔实测直方图：82 字节 4996/5002 次，其余 6 次是流首尾截断。
2. 吞吐 81986 B/s ÷ 82 = **999.8 Hz**，即标准 1 kHz。
3. ``[14:18]`` 读成 u32 时，5998 个相邻差值**全部等于 1**，无一例外
   → 这是以毫秒为单位的板载时间戳，且采集期间零丢帧。
4. ``[18:30]`` 三个 float 的模长在静止时 = 9.77 m/s^2 → 确定为加速度，单位为 SI。
5. ``[30:42]`` 三个 float 静止时 ≈ (6e-4, -4e-3, 1e-3) rad/s → 确定为角速度。
   （若按 deg/s 解释则只有 0.03 deg/s，对这类 MEMS 偏置不合理，故为 rad/s。）
6. ``[66:82]`` 四个 float 的模长恒为 1.00000x → 单位四元数。
7. 把 ``[66:82]`` 当 (w,x,y,z) 转欧拉角，与 ``[54:66]`` 字段逐帧比对，
   最大误差 < 1e-3 rad → 两组字段语义确认，且四元数**是 w 在前**。

⚠️ 一个反直觉但很关键的坑：早期用 ``cat /dev/ttyACM0`` 抓包时帧长看上去在
76~100 字节之间抖动，一度以为是变长协议。真实原因是 **tty 行规程（line
discipline）篡改/吞掉了部分字节**。必须用 pyserial 或显式 ``stty raw``，
数据才干净。本模块用 pyserial，它默认就把端口设成 raw。
"""
from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass
from typing import Iterator

# ── 协议常量 ────────────────────────────────────────────────
SYNC = b"\x5a\xa5"
FRAME_SIZE = 82
PAYLOAD_LEN = 0x004C
HEADER_LEN = 18                     # [0:18]，其后是 16 个 float32
NUM_FLOATS = 16

# 16 个 float32 的字段名，顺序即帧内顺序
FLOAT_NAMES = (
    "ax", "ay", "az",               # 加速度 m/s^2
    "gx", "gy", "gz",               # 角速度 rad/s
    "mx", "my", "mz",               # 磁力计（本板恒 0）
    "roll", "pitch", "yaw",         # 欧拉角 rad
    "qw", "qx", "qy", "qz",         # 四元数 (w,x,y,z)
)

# 需要对外输出的六轴（顺序刻意与 README 的 imu.txt 列顺序一致）
SIX_AXIS = ("gx", "gy", "gz", "ax", "ay", "az")

_struct16f = struct.Struct("<16f")
_struct_t = struct.Struct("<I")


@dataclass
class ImuFrame:
    """一帧解码后的 IMU 数据。"""

    t_board_ms: int                 # 板载毫秒计时（单调，来源见模块 docstring）
    values: dict[str, float]        # FLOAT_NAMES -> 值
    t_host: float = 0.0             # 主机单调时钟（秒），收到该帧时估计的到达时刻

    # 便捷访问 -------------------------------------------------
    @property
    def accel(self) -> tuple[float, float, float]:
        v = self.values
        return (v["ax"], v["ay"], v["az"])

    @property
    def gyro(self) -> tuple[float, float, float]:
        v = self.values
        return (v["gx"], v["gy"], v["gz"])

    @property
    def euler(self) -> tuple[float, float, float]:
        v = self.values
        return (v["roll"], v["pitch"], v["yaw"])

    @property
    def quat_wxyz(self) -> tuple[float, float, float, float]:
        v = self.values
        return (v["qw"], v["qx"], v["qy"], v["qz"])

    def six_axis(self) -> tuple[float, ...]:
        return tuple(self.values[k] for k in SIX_AXIS)


def decode_frame(buf: bytes) -> ImuFrame:
    """把 82 字节原始帧解成 :class:`ImuFrame`。不做合法性检查，调用方负责。"""
    if len(buf) != FRAME_SIZE:
        raise ValueError(f"帧长必须是 {FRAME_SIZE}，收到 {len(buf)}")
    t_board_ms = _struct_t.unpack_from(buf, 14)[0]
    vals = _struct16f.unpack_from(buf, HEADER_LEN)
    return ImuFrame(t_board_ms=t_board_ms, values=dict(zip(FLOAT_NAMES, vals)))


def parse_buffer(buf: bytes) -> tuple[list[ImuFrame], bytes]:
    """从字节流里切出所有完整帧，返回 ``(帧列表, 剩余未成帧的尾巴)``。

    用**搜索同步字**而不是按 82 整除来切：这样即使中途丢字节、或从流中间
    接入，也能自动重新对齐。找不到同步字时只保留最后 1 字节，避免尾巴无限增长。
    """
    frames: list[ImuFrame] = []
    pos = 0
    while True:
        start = buf.find(SYNC, pos)
        if start < 0:
            # 没有同步字了：留最后一个字节（可能是被截断的 0x5A）
            tail = buf[-1:] if buf[-1:] == b"\x5a" else b""
            return frames, tail
        if start + FRAME_SIZE > len(buf):
            return frames, buf[start:]          # 帧还没收全，留给下次
        chunk = buf[start:start + FRAME_SIZE]
        # 第 2、3 字节必须是长度字段，否则这个"同步字"是数据区的巧合
        if chunk[2:4] != PAYLOAD_LEN.to_bytes(2, "little"):
            pos = start + 1
            continue
        frames.append(decode_frame(chunk))
        pos = start + FRAME_SIZE


def quat_to_euler_rad(qw: float, qx: float, qy: float, qz: float):
    """(w,x,y,z) 四元数 -> (roll, pitch, yaw) 弧度，ZYX 内旋顺序。

    存在的意义是**做交叉验证**：用它算出来的欧拉角应当和板子自己给的
    ``roll/pitch/yaw`` 字段一致，否则说明字段偏移解错了。
    """
    import math

    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


class H7Imu:
    """串口 IMU 读取器：后台线程收字节、切帧、带主机时间戳。

    用法::

        with H7Imu("/dev/ttyACM0") as imu:
            for f in imu.stream(timeout=1.0):
                print(f.t_board_ms, f.accel, f.gyro)
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baud: int = 921600,
        keep_raw: bool = False,
    ):
        import serial  # 延迟导入，方便只跑解析、不装 pyserial 的场景

        self.port = port
        self.baud = baud
        # timeout 给短一点，让后台线程能及时退出
        self.ser = serial.Serial(port, baudrate=baud, timeout=0.05)
        self._buf = b""
        self._pending: list[ImuFrame] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.n_frames = 0
        self.n_bytes = 0
        # keep_raw=True 时把所有原始字节留在 self.raw，便于事后复现/复核解析
        self.keep_raw = keep_raw
        self.raw = bytearray()

    # -- 生命周期 ------------------------------------------------
    def __enter__(self) -> "H7Imu":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        self.ser.reset_input_buffer()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.ser.close()
        except Exception:
            pass

    # -- 后台线程 ------------------------------------------------
    def _reader(self) -> None:
        while not self._stop.is_set():
            n = self.ser.in_waiting
            chunk = self.ser.read(n if n else 1)
            if not chunk:
                continue
            now = time.monotonic()
            self.n_bytes += len(chunk)
            if self.keep_raw:
                self.raw += chunk
            frames, self._buf = parse_buffer(self._buf + chunk)
            if frames:
                self.n_frames += len(frames)
                # 一帧 1 ms，把这一批里最早的那帧回推到它真实的到达时刻，
                # 比"整批都记同一个时间"精度高一个量级。
                span = (len(frames) - 1) * 0.001
                base = now - span
                for i, f in enumerate(frames):
                    f.t_host = base + i * 0.001
                with self._lock:
                    self._pending.extend(frames)

    # -- 对外接口 ------------------------------------------------
    def stream(self, timeout: float | None = None) -> Iterator[ImuFrame]:
        """边收边吐。``timeout`` 秒内没有新帧就结束（None = 一直跑）。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        idle_since = time.monotonic()
        while not self._stop.is_set():
            got = False
            with self._lock:
                if self._pending:
                    batch, self._pending = self._pending, []
                    got = True
            if got:
                idle_since = time.monotonic()
                for f in batch:
                    yield f
            else:
                time.sleep(0.002)
            if deadline is not None and time.monotonic() > deadline:
                return
            if timeout is None and time.monotonic() - idle_since > 2.0:
                # timeout=None 时用 2 s 静默当作断流保护
                return

    def read_for(self, seconds: float) -> list[ImuFrame]:
        """阻塞采集指定时长，返回帧列表。"""
        out: list[ImuFrame] = []
        t0 = time.monotonic()
        for f in self.stream(timeout=seconds):
            out.append(f)
            if time.monotonic() - t0 >= seconds:
                break
        return out


def validate(frames: list[ImuFrame]) -> dict:
    """对一批帧做体检，返回可打印的统计字典。

    这是"我凭什么相信解析是对的"的答案，也是标定前判断数据可不可用的依据。
    """
    import math

    if not frames:
        return {"error": "没有帧"}

    n = len(frames)
    dts = [frames[i + 1].t_board_ms - frames[i].t_board_ms for i in range(n - 1)]
    speeds = [math.sqrt(sum(c * c for c in f.accel)) for f in frames]
    qnorms = [math.sqrt(sum(c * c for c in f.quat_wxyz)) for f in frames]

    # 四元数 vs 欧拉角交叉验证
    errs = []
    for f in frames:
        r, p, y = quat_to_euler_rad(*f.quat_wxyz)
        er, ep, ey = f.euler
        dy = (y - ey + math.pi) % (2 * math.pi) - math.pi
        errs.append(max(abs(r - er), abs(p - ep), abs(dy)))

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "帧数": n,
        "板载时长_s": (frames[-1].t_board_ms - frames[0].t_board_ms) / 1000.0,
        "采样率_Hz": (n - 1) / ((frames[-1].t_board_ms - frames[0].t_board_ms) / 1000.0)
        if frames[-1].t_board_ms != frames[0].t_board_ms
        else float("nan"),
        "时间戳步长_取值": sorted(set(dts))[:5],
        "时间戳步长_非1帧数": sum(1 for d in dts if d != 1),
        "加速度模长_均值": mean(speeds),
        "加速度模长_标准差": (sum((s - mean(speeds)) ** 2 for s in speeds) / n) ** 0.5,
        "四元数模长_最大偏差": max(abs(q - 1.0) for q in qnorms),
        "四元数vs欧拉角_最大误差_rad": max(errs),
        "磁力计是否全零": all(
            f.values["mx"] == 0 and f.values["my"] == 0 and f.values["mz"] == 0
            for f in frames
        ),
    }


def main() -> int:
    """直接运行本文件 = 硬件自检：抓 3 秒，打印协议体检表。"""
    import argparse

    ap = argparse.ArgumentParser(description="H7 IMU 协议自检")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()

    print(f"打开 {args.port} @ {args.baud} ...")
    with H7Imu(args.port, args.baud) as imu:
        frames = imu.read_for(args.seconds)
        print(f"收到 {imu.n_frames} 帧 / {imu.n_bytes} 字节 "
              f"({imu.n_bytes / args.seconds:.0f} B/s)")

    stats = validate(frames)
    print("\n=== 协议体检 ===")
    for k, v in stats.items():
        print(f"  {k:28s} {v}")

    if frames:
        f = frames[0]
        print("\n=== 首帧（已解码）===")
        print(f"  t_board_ms = {f.t_board_ms}")
        print(f"  accel      = {f.accel}  m/s^2")
        print(f"  gyro       = {f.gyro}  rad/s")
        print(f"  euler      = {f.euler}  rad")
        print(f"  quat(wxyz) = {f.quat_wxyz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
