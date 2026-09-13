# Camera-IMU 联合标定

RoboWalker 算法组 2027 新生提前批大作业 · 任务 4 实现。

对机械固连的相机与 IMU，完成 **相机内参标定** 与 **Camera-IMU 外参标定**，
并给出可复现的误差验证。

---

## 1. 使用环境

实测通过的运行环境（本机原生 Ubuntu，**不是 WSL**）：

| 项目 | 版本 / 说明 |
|---|---|
| 操作系统 | Ubuntu 24.04.4 LTS (Noble), 内核 6.17 |
| Python | 3.12.3 |
| 虚拟环境 | `~/calib_ws/.venv`（`--system-site-packages`，可同时用系统 numpy/scipy） |
| 相机 | ASUS FHD webcam（UVC），`/dev/video0`，640×480@30fps |
| IMU | 串口 IMU（`/dev/ttyUSB*` 或 `/dev/ttyACM*`），协议可通过正则适配 |
| 是否需要 ROS | **不需要**。本任务是纯离线标定，不依赖 ROS 2 |

## 2. 依赖库

| 库 | 版本 | 用途 |
|---|---|---|
| numpy | 1.26.4 | 数值计算 |
| scipy | 1.11.4 | 优化（`least_squares`）、旋转表示（`Rotation`） |
| opencv-python | 4.11.0.86 | 棋盘格/ArUco 检测、`calibrateCamera`、`solvePnP`、去畸变 |
| matplotlib | 3.6.3 | 误差曲线与可视化报告 |
| PyYAML | 6.0.1 | 配置文件与结果文件读写 |
| pyserial | 3.5 | 串口 IMU 采集 |

依赖清单见 `requirements.txt`。

## 3. 安装方法

本机的虚拟环境**已经建好**，直接用即可：

```bash
cd ~/calib_ws
source .venv/bin/activate      # 或始终用 .venv/bin/python 显式调用
```

在**另一台机器**上从零复现：

```bash
# Ubuntu 24.04 系统 python 受 PEP 668 保护，且未预装 pip/python3-venv。
# 若有 sudo，最短路径是：
sudo apt install -y python3-venv python3-pip v4l-utils
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt

# 若没有 sudo（本机就是这种情况），用官方 zipapp 自举，完全不碰系统 Python：
curl -sSLo pip.pyz https://bootstrap.pypa.io/pip/pip.pyz
python3 pip.pyz install --target .bootstrap virtualenv
PYTHONPATH=.bootstrap python3 -m virtualenv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 4. 程序运行方法

```bash
cd ~/calib_ws

# 第 0 步：硬件自检（相机、串口、依赖一次查完）
.venv/bin/python tools/probe_hardware.py

# 第 1 步：IMU 协议自检（抓 3 秒，验证帧格式/采样率/四元数自洽）
python3 calib/imu_h7.py --port /dev/ttyACM0 --seconds 3

# 第 2 步：只采 IMU（不接相机，最常用来确认硬件还活着）
.venv/bin/python tools/capture_h7.py --imu-only --seconds 5 --out data/probe

# 第 3 步：采集数据。按键 s=存帧 a=自动连拍 q=结束
.venv/bin/python tools/capture_h7.py \
    --out data/session_01 --camera 0 --width 640 --height 480 \
    --serial /dev/ttyACM0 --lock-exposure

# 第 4 步：运动激励体检（判断这组数据够不够解外参）
.venv/bin/python tools/check_motion.py --imu data/session_01/imu.txt

# 第 5 步：标定（待实现）
.venv/bin/python calib/run_calibration.py --config config/calibration.yaml

# 无窗口自测（验证整条链路，不弹窗）
.venv/bin/python tools/capture_h7.py --out /tmp/selftest --frames 3
```

> 旧工具 `tools/capture_dataset.py` 面向**文本协议**的 IMU（维特 ASCII 等），
> 与当前这块二进制 H7 板不兼容，保留仅作参考。当前硬件请用 `tools/capture_h7.py`。


### 采集时的动作要求

标定精度**主要取决于采集动作的质量**，而不是算法：

1. 开头保持 **2 秒完全静止**（用于零偏标定与重力方向对齐）；
2. 缓慢绕 **x / y / z 三个轴各自转动**若干圈（只绕一个轴转是手眼标定无解的头号原因）；
3. 手持做 **小幅平移**（纯旋转无法确定平移外参）；
4. 让标定板在图像里走遍 **画面四角与中心**、并覆盖不同 **倾角**；
5. 标定板与相机 IMU 之间保持**机械固连**，全程不要碰到相机与 IMU 的相对位置。

## 5. 输入数据格式

### 5.1 图像与时间戳

```text
data/session_01/images/000000.png, 000001.png, ...
data/session_01/image_timestamps.txt
```

```text
# filename timestamp_seconds (monotonic clock, unit: s)
000000.png 7011.193130496
000001.png 7011.401313845
```

时间戳统一使用**本机单调时钟**（`time.monotonic()`）。原因：系统时钟会被 NTP 校时跳变，
而单调时钟不会倒退，跨设备对齐更可靠。

### 5.2 IMU 数据

```text
data/session_01/imu.txt
```

```text
# timestamp gx gy gz ax ay az   (s, rad/s ×3, m/s^2 ×3)
7011.190000000  0.0012 -0.0003 0.0007  0.021 -0.013 9.803
```

单位约定（程序内部统一）：

| 量 | 单位 |
|---|---|
| 时间戳 | 秒（文件里若是 ms/us/ns，用 `config.time.unit` 声明） |
| 角速度 | rad/s |
| 加速度 | m/s² |

串口原始协议如果不是 SI 单位，用 `--imu-scale deg/s,g` 让采集程序做换算。

### 5.3 标定板参数

在 `config/calibration.yaml` 的 `target` 段声明：

```yaml
target:
  type: chessboard
  chessboard:
    pattern_size: [9, 6]     # 内角点数（列 × 行），不是方格数
    square_size: 0.025       # 单位：米，必须与实物一致
```

`square_size` 若与实物不符，**相机内参仍然正确，但平移外参的尺度会整体错**。

## 6. 输出数据格式

标定结果同时写 `results/calibration_result.yaml` 与 `.json`，可被程序重新读取；
每次运行另存一份带时间戳的快照，用于"多次独立标定结果对比"。

```yaml
camera:
  model: pinhole
  image_size: [640, 480]
  intrinsics:
    fx: 0.0
    fy: 0.0
    cx: 0.0
    cy: 0.0
  distortion:
    model: plumb_bob
    coefficients: [k1, k2, p1, p2, k3]
  reprojection_error_rms: 0.0      # 单位：像素
extrinsic:
  convention: T_IC_means_camera_to_imu
  T_IC:                            # 4×4 行主序，见第 7 节坐标系定义
    - [r00, r01, r02, tx]
    - [r10, r11, r12, ty]
    - [r20, r21, r22, tz]
    - [0.0, 0.0, 0.0, 1.0]
  rotation_euler_deg: [roll, pitch, yaw]
  translation_m: [tx, ty, tz]
validation:
  reprojection_error_px: 0.0
  motion_consistency_deg: 0.0
  cross_validation: []
```

## 7. 坐标系定义（**必须明确**，任务文档 3.2 与验收第 8 条要求）

| 坐标系 | 记法 | 定义 |
|---|---|---|
| 相机系 | `C` | 原点在相机光心；x 向右，y 向下，z 沿光轴向前（OpenCV 约定） |
| IMU 系 | `I` | 原点在 IMU 测量中心；三轴方向以 IMU 丝印/数据手册为准 |
| 标定板系 | `B` | 原点在棋盘格左上角第一个内角点（ArUco 时为指定 marker 中心）；z 垂直板面向外 |

**外参约定：**

$$
T_{IC} = \begin{bmatrix} R_{IC} & t_{IC} \\ 0 & 1 \end{bmatrix}
$$

表示 **把相机坐标系 `C` 下的一个点变换到 IMU 坐标系 `I` 下**：

$$
p_{I} = R_{IC}\, p_{C} + t_{IC}
$$

即 **`T_IC` = 相机系 → IMU 系的变换**；它的逆 `T_CI = T_IC^{-1}` 是 IMU 系 → 相机系。

> ⚠️ 这一约定是**任意的**，业界两种写法都有人用（Kalibr 输出的是 `T_cam_imu`，语义为
> IMU 系 → 相机系，和本文件相反）。因此本程序在配置文件里用
> `extrinsic.convention` 显式声明，并在结果文件里原样回写，避免验收时产生歧义。

## 8. 结果验证

任务文档要求至少一种，本程序默认全做：

| 方式 | 指标 | 合格参考 |
|---|---|---|
| 重投影误差 | 所有角点在所有图像上的 RMS 误差 | < 0.5 px（好）/ < 1.0 px（可接受） |
| 标定残差 | 内参标定的平均重投影残差 | 与上同源，用于判断内参是否收敛 |
| 运动一致性 | 由 Camera-IMU 各自推算的姿态差角 | 均值 < 1-2° |
| 交叉验证 | 数据切 N 份独立标定，比较外参离散度 | 旋转标准差 < 1°，平移标准差 < 数 mm |
| 与成熟工具对比 | 与 Kalibr 等结果比对外参差角/距离 | 差角 < 1-2° |

判断"结果是否合理"的依据会写进 `results/report/report.md`：指标值 + 是否达到上表阈值 + 不达标时的可能原因。

## 9. 目录结构

```text
calib_ws/
├── README.md                  ← 本文件
├── requirements.txt
├── config/
│   └── calibration.yaml        ← 所有可调参数（不硬编码在代码里）
├── calib/                      ← 标定源码
│   ├── imu_h7.py               ✅ H7 IMU 二进制协议解析（实测验证，见 doc/H7_IMU协议.md）
│   ├── io_data.py              数据读取（图像/时间戳/IMU）
│   ├── target_detect.py        标定板检测（棋盘格/ArUco/AprilTag）
│   ├── intrinsics.py           相机内参标定
│   ├── imu.py                  IMU 预处理、零偏、姿态积分
│   ├── sync.py                 相机-IMU 时间对齐
│   ├── extrinsics.py           Camera-IMU 外参（手眼标定 AX=XB）
│   ├── validate.py             误差与一致性验证
│   └── run_calibration.py      主入口
├── tools/
│   ├── probe_hardware.py       硬件与权限自检
│   ├── capture_h7.py           ✅ 数据采集（UVC 相机 + H7 IMU，当前硬件用这个）
│   ├── capture_dataset.py      数据采集（旧版，只支持文本协议 IMU，保留参考）
│   ├── check_motion.py         运动激励体检
│   └── setup_permissions.sh    udev/用户组授权（需 sudo）
├── doc/
│   ├── H7_IMU协议.md           ✅ 协议逆向结论与证据链
│   ├── 相机取流诊断.md          海康 U3V 取流诊断
│   ├── IMU板排错清单.md
│   └── ROS2上手说明.md
├── data/                       采集数据（示例数据另附）
└── results/                    标定结果与验证报告
```

## 10. 当前进度

| 步骤 | 状态 |
|---|---|
| 环境搭建（Python 依赖、虚拟环境） | ✅ 完成 |
| 相机接入与画面验证 | ✅ 笔记本 UVC 相机可用（`/dev/video0` 640×480@30fps）；海康 U3V 工业相机**尚未取流成功**，见 doc/相机取流诊断.md |
| 串口 IMU 接入与协议解析 | ✅ 完成（H7 板 82 字节帧 @1 kHz 全部字段已解出并验证，见 doc/H7_IMU协议.md） |
| 数据采集工具 | ✅ 完成（`tools/capture_h7.py`，含无窗口自测） |
| 运动激励体检工具 | ✅ 完成 |
| 标定算法（`calib/`） | ⏳ 待实现（目录目前只有 `imu_h7.py`） |
| 验证与报告 | ⏳ 待实现 |

## 11. 已知限制

- 本机 `sudo` 需要密码，因此**设备授权脚本还没跑过**。目前 `/dev/ttyACM0` 能用
  是因为 udev 兜底规则给了 `ttyACM*` 0666；但 **`/dev/rw_imu` 这个稳定别名还不存在**
  （规则里没有新板的 PID）。要修好请执行一次：
  `sudo bash ~/calib_ws/tools/setup_permissions.sh`（脚本已更新，含 `0483:6666`）。
- 当前无 `v4l-utils`，摄像头参数查看依赖 OpenCV 而非 `v4l2-ctl`。
- 相机的曝光时间若为自动模式，帧率会有抖动，高动态运动下时间对齐误差变大；
  正式采集请加 `--lock-exposure`（已加入采集脚本），并在结果里核对实际帧率。
- **IMU 磁力计恒为 0**（未启用）→ yaw 不可观测，姿态只能靠陀螺积分，会缓慢漂移。
  写外参标定程序时不要依赖磁力计通道。
- 海康 U3V 相机在 aravis 下的**取流仍不稳定**：枚举与 GenICam 寄存器读写时好时坏，
  取流阶段报 `USB3Vision write_memory timeout`。已定性为 USB 链路/协议交互问题
  （非权限），建议优先试换 USB3 口与线缆，并走海康 MVS SDK 路线。

---

## 12. 补充：本机实际硬件（2026-09-10 实测）

| 设备 | 型号 / 标识 | 连接方式 | 状态 |
|---|---|---|---|
| 工业相机 | 海康机器人 **MV-CS020-10UC** (SN `DA3489519`) | USB 3.2 Gen1，5 Gbps，800 mA | 枚举正常，**取流未打通**（aravis 报 USB 超时） |
| IMU 板（在用） | **H7_IMU_With_EKF** (SN `375939523233`, VID:PID `0483:6666`) | USB-C → CDC 虚拟串口 `/dev/ttyACM0` | ✅ **完全正常**：82 字节帧 @1 kHz，六轴+欧拉角+四元数，1 秒零丢帧 |
| IMU 板（已退役） | STM32 Virtual ComPort (SN `3144366B3233`, VID:PID `0483:5740`) | CDC `/dev/ttyACM0` | ❌ 数据区恒为 0，已换成上面那块 |
| 笔记本摄像头 | ASUS FHD webcam / IR camera | UVC `/dev/video0-3` | ✅ 可用，算法开发阶段用它 |

> ⚠️ **2026-09-13 硬件变更**：旧 IMU 板（`0483:5740`，数据恒 0）已换成
> **H7_IMU_With_EKF（`0483:6666`）**，新板数据完全正常。
> 下文 12.1 节保留的是旧板的故障诊断记录，**仅作考古，不要照着它判断当前状态**。
> 当前板的协议见 [doc/H7_IMU协议.md](doc/H7_IMU协议.md)。

### 相机取流路线（不兼容 UVC，不能用 `--camera` 参数）

`lsusb -v` 显示该相机三个接口全是 `bInterfaceClass=239 / bInterfaceSubClass=5 (USB3 Vision)`，
**没有 UVC Video 类接口**，所以 `/dev/video*` 与 OpenCV 均无法打开它。可选方案：

| 方案 | 依赖 | 优点 | 缺点 |
|---|---|---|---|
| A. aravis | ✅ **已装**（`aravis-tools` / `libaravis-0.8-0` 0.8.30，科大源） | 开源、免注册 | 对本相机**取流不稳定**，见下 |
| B. 海康 MVS SDK（**当前首选**） | 官网下载 `MVS-x.x.x_x86_64_xxxxxx.tar.gz`（需注册账号） | 官方支持、有硬件时间戳、参数最全 | 注册下载、体积大、闭源 |
| C. harvesters + GenTL | 已装 `harvesters 1.4.3` + `genicam 1.6.0`，但**缺 .cti producer** | Python 接口现成 | producer 必须来自 A 或 B |

**aravis 现状（2026-09-13 复测，比 9/10 有进展）**：设备能被枚举，GenICam 属性
（`SensorSize` / `Gain` / `ExposureTime`）也能读出来，但**取流阶段失败**：

```
Found 1 device
Testing 'U3V:MV-CS020-10UC'
Properties:SensorSizeReadout        SUCCESS
Properties:ExposureTimeReadout      SUCCESS
MultipleAcquisitionA:BufferCheck    FAILURE 0/10 [AcquisitionStop] USB3Vision write_memory timeout
```

且**行为不稳定**：同一分钟内重跑，会退回到 `Failed to connect`。
寄存器读写时好时坏 + 取流超时，典型指向 **USB3 链路质量问题（线缆/供电/主控口）**，
而不是纯软件不兼容。

**建议的排查顺序**（成本从低到高）：

1. **换线**：用相机原装 USB3 线，或明确标注 5 Gbps 的短线；很多"随机超时"就是线。
2. **换口**：直接插机身上的 USB3 口，不要经过扩展坞/Hub。
3. 确认链路速率：`lsusb -t` 里该设备应显示 `5000M`（当前是，若变 480M 就是线/口退化）。
4. 以上都排完仍失败，再上 **MVS SDK**（官方路线，且带硬件时间戳，
   对 Camera-IMU 标定的时间对齐有实质收益）。

```bash
lsusb -t | grep -B1 -A3 2bdf      # 看链路速率是否为 5000M
timeout 30 arv-test-0.8           # 看 GenICam 属性与取流各自成败
```

### 权限

`/dev/ttyACM0` 默认 `root:dialout 660` 且**没有 ACL**，当前用户不在 `dialout` 组。
一次性修复（需要 sudo 密码，在用户自己的终端执行）：

```bash
sudo bash ~/calib_ws/tools/setup_permissions.sh
```

会加入 `dialout`/`plugdev`/`video` 组、写入 udev 规则（覆盖 STM32/CH340/CP210x/FTDI/海康 U3V），
并给 STM32 一个稳定别名 `/dev/rw_imu`。


### 12.1 旧 IMU 板数据流诊断结论（2026-09-10 实测，⚠️ 已随硬件更换失效）

> **这块板已经不用了，本节结论对当前硬件无效。**
> 保留它是因为记录了一次完整的「设备连上了但数据不动」排查过程，
> **方法论仍然可复用**（先证明不丢包 → 再证明是二进制 → 再证明帧长稳定 →
> 最后逐字节统计确认数据区是否在更新）。
>
> 当前在用板的协议与验证见 [doc/H7_IMU协议.md](doc/H7_IMU协议.md)。
> 旧板症状：28 字节定长帧 @1 kHz，但 8045 帧逐字节完全相同，数据区恒 0。

<!-- 以下为旧板原始记录，仅作考古 -->

**现象**：设备以确定性的 **28 字节定长帧 @ 1 kHz**（28,000 B/s）持续发送，
但**全部 8045 帧逐字节完全相同**，数据区恒为 0。

```
abcd 0000 713d aabf | 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000
└── 8 字节帧头/常量 ──┘ └──────────── 20 字节数据区，恒为 0 ────────────┘
```

证据链（三条独立测量互相印证，排除了采集端问题）：

| 检查 | 方法 | 结果 |
|---|---|---|
| 是否丢包造成假象 | 重新采集 15 s，对比吞吐与帧头计数 | 421,888 B ÷ 28 = 15,067 帧 vs 帧头计数 15,068 → **无丢包** |
| 是否协议是文本 | 可打印字符占比 | 7.1% → **二进制协议** |
| 帧长是否稳定 | 自相关找最短重复周期 | 28 字节，**一致率 100.00%** |
| 数据区是否在更新 | 逐字节位置的取值种类数 | **0/28 个字节有变化 → 传感器未被采样** |

**结论**：固件主循环在跑（USB CDC 持续发送、蜂鸣器按周期鸣响），
但**传感器数据从未被写入帧缓冲区**。最可能是固件自检阶段读 IMU 失败后
进入「带蜂鸣提示的错误循环」，或传感器本身未应答。

**检测工具**：`tools/analyze_frames.py` —— 自动识别定长帧周期、逐字节统计取值种类、
并尝试按 int16/uint16/float32 解码变化区域。用途是**验证重新烧录固件后数据区是否开通**：

```bash
.venv/bin/python tools/analyze_frames.py --port /dev/rw_imu --seconds 5
```

- 输出「✅ 数据区在持续变化」= 固件正常，可以开始标定；
- 输出「❌ 全部帧完全相同」= 仍是本例的故障状态。

**这不是标定算法的问题**：即使拿到这样的数据流，任何标定程序都无法解出外参，
因为 IMU 的 6 个通道（3 轴陀螺 + 3 轴加速度）在全时段内是常量。

### 离线自测

不接硬件也能验证整条解析链路（用伪终端伪造 IMU）：

```bash
.venv/bin/python tools/selftest_serial.py                    # 维特 ASCII 格式
.venv/bin/python tools/selftest_serial.py --format csv       # 数字列格式
.venv/bin/python tools/selftest_serial.py --algo-deg         # 验证 deg/s、g 的单位换算
```
