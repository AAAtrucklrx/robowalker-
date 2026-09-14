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

### 4.1 真实数据：两条命令出结果

```bash
cd ~/calib_ws

# ① 硬件自检（30 秒）
python3 calib/imu_h7.py --port /dev/ttyACM0 --seconds 3
python3 calib/camera.py --source hik --grab 3 --out /tmp/cam

# ② 采集（工具会实时提示角速度与深度跨度，不合格当场红色警告）
.venv/bin/python tools/capture_h7.py --out data/session_01 \
    --camera-source hik --width 1624 --height 1240 --pixel-format Mono8 \
    --exposure 200000 --gain 20 --frame-rate 4 \
    --pattern 8x5 --square-mm 42 --serial /dev/ttyACM0

# ③ 标定：一条命令跑完 C1→C2→C3→C4，自动生成报告
.venv/bin/python calib/run_calibration.py --session data/session_01 \
    --pattern 8x5 --square-mm 42

# 只看内参（快，用来先判断数据质量）
.venv/bin/python calib/run_calibration.py --session data/session_01 \
    --pattern 8x5 --square-mm 42 --only-intrinsics
```

> `--pattern` 填**内角点数**，不是方格数：9×6 方格 → `8x5`。
> 板子怎么选、印多大、可用距离区间见 **[doc/标定板选择指南.md](doc/标定板选择指南.md)**。

### 4.2 验证与对照

```bash
# 合成自检（不依赖任何硬件，证明算法链路正确）
.venv/bin/python tools/selftest_intrinsics.py        # C1：点级 + 图像级两层
.venv/bin/python tools/selftest_pipeline.py --step 2e-3   # C1+C2+C3 五项

# 采集工具离线整链路自测（伪造 H7 串口 + 真实 UVC 相机）
.venv/bin/python tools/selftest_capture.py

# 与成熟工具对比（ROS camera_calibration，免 sudo）
#   抽包位置在项目内 third_party/roscc（**不是 /tmp** —— /tmp 被清理后这项
#   验证会静默失效），.deb 已随仓库缓存，断网也能重建。
bash tools/setup_ros_bypass.sh
.venv/bin/python tools/compare_with_ros.py --session data/session_01 \
    --pattern 11x8 --square-mm 20
#   合成数据（带 truth.json）会自动多出一节「与真值对比」，用于裁决两套实现
#   谁更接近真值 —— 只比"两者差多少"是无法判定谁对的。

# 受控实验：深度多样性对内参可辨识性的影响
.venv/bin/python tools/experiment_depth_diversity.py

# 造一份落盘的合成数据集（用于端到端验证交付物本身）
.venv/bin/python tools/make_synthetic_session.py --out data/synth_01
```

### 4.3 采集时的动作要求

标定精度**主要取决于采集动作的质量**，而不是算法。`capture_h7.py` 会实时盯着下面第 2、4 条：

1. 开头保持 **3 秒完全静止**（零偏标定与重力对齐**只能**靠静止段，没有这段后面全废）；
2. 绕 **x / y / z 三个轴各自快速转动**（只绕一个轴 → 手眼标定数学上退化）。
   **目标角速度 ≥ 90 °/s** —— 实测 `τ 分辨率 ≈ 角度残差 / 角速度`，转慢了时间对齐必不准。
   工具会在 < 40 °/s 时红色警告；
3. 手持做 **小幅平移**（纯旋转无法确定平移外参）；
4. 让标定板走遍**画面四角与中心**、覆盖不同倾角，并且**走遍近 / 中 / 远三档距离**。
   **深度跨度必须 ≥ 2×** —— 受控实验证明跨度 1.0× 时 `fx` 偏 **+4.17%**，
   而重投影误差只有 0.13 px **完全不报警**。工具会实时显示 `depth=N.Nx`；
5. 标定板与相机 IMU 之间保持**机械固连**，全程不要碰到相对位置
   （USB 线一定要做**应力释放**，别让线拽着 IMU）。
6. **打印后必须量一下板子外框**，确认没有缩放（打印时不要勾"适应页面"）。
   `square_size` 是唯一的度量尺度基准，而**尺度错了几乎不报警**：实测把 20 mm
   报成 40 mm（2 倍错误），内参/畸变/`R_IC`/`τ` **逐位不变**，`|t_IC|` 也只从
   5.03 变到 5.41 cm（**+7.5%，不是翻倍**），唯一报警是杠杆臂 LS 残差
   （0.093 → 0.114 m/s²）。**别只量一格** —— 量一格差 1 mm 就是 5% 尺度误差。
   本板（12×9 格 × 20 mm）外框应为 **240 × 180 mm**。
7. **曝光必须合适**（`capture_h7.py` 会实时判并报警）。
   判据是**饱和像素占比**而不是均值 —— 棋盘格黑白各半，均值天然居中，
   过曝帧的均值可能只是个"看起来正常"的 225。过曝会让白格与背景一起顶到
   255，角点一个都检不出。**2026-09-14 第一次真机试采集整场报废就是这个原因。**
   建议先用 `tools/hunt_board.py --exposure 15000 --gain 10` 试拍确认。

> 旧工具 `tools/capture_dataset.py` 面向**文本协议**的 IMU（维特 ASCII 等），
> 与当前这块二进制 H7 板不兼容，保留仅作参考。当前硬件请用 `tools/capture_h7.py`。

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
│   ├── imu_h7.py               ✅ H7 IMU 二进制协议解析（实测验证）
│   ├── camera.py               ✅ 统一相机层：U3V(aravis) + UVC(OpenCV)，双时间戳
│   ├── io_data.py              ✅ 数据集与结果读写（YAML/JSON 可回读）
│   ├── target_detect.py        ✅ 标定板检测（棋盘格 SB 检测器 / ChArUco）
│   ├── intrinsics.py           ✅ C1 相机内参标定 + train/val 分离验证
│   ├── sync.py                 ⏳ C2 相机-IMU 时间对齐
│   ├── extrinsics.py           ⏳ C3 Camera-IMU 外参（手眼标定 AX=XB）
│   ├── validate.py             ⏳ C4 误差与一致性验证
│   └── run_calibration.py      ⏳ 主入口
├── tools/
│   ├── make_board.py           ✅ 生成尺寸精确的可打印标定板（PDF，带校验尺）
│   ├── selftest_intrinsics.py  ✅ C1 两层合成自检（不需要实物标定板）
│   ├── capture_h7.py           ✅ 数据采集（相机 + H7 IMU，含无窗口自测）
│   ├── probe_hardware.py       硬件与权限自检
│   ├── capture_dataset.py      旧版采集（只支持文本协议 IMU，保留参考）
│   ├── check_motion.py         运动激励体检
│   └── setup_permissions.sh    udev/用户组授权（需 sudo）
├── boards/                     生成的标定板 PDF/PNG（打印用）
├── doc/
│   ├── H7_IMU协议.md           ✅ 协议逆向结论与证据链
│   ├── 相机取流诊断.md          ✅ 海康 U3V 取流诊断（结论：是 USB 口的问题）
│   ├── IMU板排错清单.md
│   └── ROS2上手说明.md
├── data/                       采集数据（示例数据另附）
└── results/                    标定结果与验证报告
```

## 10. 当前进度

| 步骤 | 状态 |
|---|---|
| 环境搭建（Python 依赖、虚拟环境） | ✅ 完成 |
| **海康 U3V 工业相机取流** | ✅ **已打通**（换 USB3 口后；aravis Python 绑定，1624×1240 Mono8 @10Hz，零丢包，**硬件时间戳可用**） |
| 笔记本 UVC 相机 | ✅ 可用（`/dev/video0` 640×480@30fps），算法开发阶段的默认数据源 |
| 串口 IMU 接入与协议解析 | ✅ 完成（H7 板 82 字节帧 @1 kHz 全部字段已解出并验证） |
| 统一相机抽象层 `calib/camera.py` | ✅ 完成（U3V + UVC，同一接口，带双时间戳） |
| 可打印标定板生成 `tools/make_board.py` | ✅ 完成（PDF 页面尺寸精确到 0.01 mm，附 100 mm 校验尺） |
| 数据采集工具 | ✅ 完成（`tools/capture_h7.py`，含无窗口自测） |
| 运动激励体检工具 | ✅ 完成 |
| **C1 相机内参** | ✅ **代码完成并通过两层合成自检**（`target_detect.py` + `intrinsics.py` + `tools/selftest_intrinsics.py`） |
| C2 时间对齐 | ⏳ 待实现（`calib/sync.py`） |
| C3 Camera-IMU 外参 | ⏳ 待实现（`calib/extrinsics.py`） |
| C4 验证与报告 | ⏳ 待实现（`calib/validate.py`、`run_calibration.py`） |

### C1 自检结果（可复现）

```bash
.venv/bin/python tools/selftest_intrinsics.py
```

| 层级 | 内容 | 结果 |
|---|---|---|
| Level 1 · 点级 | 正向投影角点 → 直接标定，**不经过图像**，检验标定数学 | ✅ fx/fy/cx/cy 误差 < 0.001 px；畸变函数相对误差 0.000% |
| Level 2 · 图像级 | 渲染棋盘格 → 检测 → 标定，端到端 | ✅ 40/40 检出；残差 0.082 px（渲染器边缘模型地板） |

## 11. 已知限制

- 本机 `sudo` 需要密码，因此**设备授权脚本还没跑过**。目前 `/dev/ttyACM0` 能用
  是因为 udev 兜底规则给了 `ttyACM*` 0666；但 **`/dev/rw_imu` 这个稳定别名还不存在**
  （规则里没有新板的 PID）。要修好请执行一次：
  `sudo bash ~/calib_ws/tools/setup_permissions.sh`（脚本已更新，含 `0483:6666`）。
- **U3V 相机对 USB 口非常敏感**：换口前 aravis 报 `USB3Vision write_memory timeout`，
  换到机身原生 USB3 口后立刻正常。**遇到"时好时坏"先怀疑线/口，不要先怀疑库。**
- aravis 的 Python 绑定（`gir1.2-aravis-0.8`，29.7 kB）**本机已安装**，可以直接用
  `calib/camera.py`。换机器时需要 `sudo apt install gir1.2-aravis-0.8`；
  没有 sudo 时的临时方案见 `calib/camera.py` 的 docstring。
- **C3 平移外参 `t_IC`（杠杆臂）当前不可用** —— 已定位到数值原因：
  方程本身精确正确（真值代入残差 4e-15），但 `|b| = 0.059 m/s²` 是
  `|R_WI·a_meas| = 9.82` 相减后剩下的量，**160:1 的抵消**；而 `a_WC` 靠对
  PnP 位置做二阶数值微分得到，平面目标的深度噪声被放大后淹没了信号
  （LS 残差 11.1 m/s² ≈ g）。见 `doc/项目状态汇报_Camera-IMU标定.md` §4.1。
  三条出路：① 尺子量 + 声明不确定度（保底）；② 相机帧率提到 50~100 Hz
  并做剧烈旋转激励；③ 上 Kalibr 做全批量优化。
- 相机曝光/增益**必须手动锁定**（`calib/camera.py` 已默认关掉自动模式）；
  自动曝光会让帧率抖动，时间对齐误差变大。
- 当前无 `v4l-utils`，UVC 参数查看依赖 OpenCV 而非 `v4l2-ctl`。
- **IMU 磁力计恒为 0**（未启用）→ yaw 不可观测，姿态只能靠陀螺积分，会缓慢漂移。
  写外参标定程序时不要依赖磁力计通道。
- 标定精度**主要取决于采集动作**，不是算法。见第 4 节的采集要求。

---

## 12. 补充：本机实际硬件（2026-09-10 实测）

| 设备 | 型号 / 标识 | 连接方式 | 状态 |
|---|---|---|---|
| 工业相机 | 海康机器人 **MV-CS020-10UC** (SN `DA3489519`) | USB 3.2 Gen1，5 Gbps，800 mA | ✅ **取流已打通**（换 USB3 口后；1624×1240 Mono8 @10Hz 零丢包，`DeviceTimestamp` 可用） |
| IMU 板（在用） | **H7_IMU_With_EKF** (SN `375939523233`, VID:PID `0483:6666`) | USB-C → CDC 虚拟串口 `/dev/ttyACM0` | ✅ **完全正常**：82 字节帧 @1 kHz，六轴+欧拉角+四元数，1 秒零丢帧 |
| IMU 板（已退役） | STM32 Virtual ComPort (SN `3144366B3233`, VID:PID `0483:5740`) | CDC `/dev/ttyACM0` | ❌ 数据区恒为 0，已换成上面那块 |
| 笔记本摄像头 | ASUS FHD webcam / IR camera | UVC `/dev/video0-3` | ✅ 可用，算法开发阶段用它 |

> ⚠️ **2026-09-13 硬件变更**：旧 IMU 板（`0483:5740`，数据恒 0）已换成
> **H7_IMU_With_EKF（`0483:6666`）**，新板数据完全正常。
> 下文 12.1 节保留的是旧板的故障诊断记录，**仅作考古，不要照着它判断当前状态**。
> 当前板的协议见 [doc/H7_IMU协议.md](doc/H7_IMU协议.md)。

### 相机取流：已打通（2026-09-13）

`lsusb -v` 显示该相机三个接口全是 `bInterfaceClass=239 / bInterfaceSubClass=5 (USB3 Vision)`，
**没有 UVC Video 类接口**，所以 `/dev/video*` 与 OpenCV **永远打不开它**——这是协议不同，
不是配置问题。实际可用的路线：

| 方案 | 状态 | 说明 |
|---|---|---|
| **A. aravis + Python 绑定** | ✅ **当前使用** | `aravis-tools` + `libaravis-0.8-0` 已有；再装 29.7 kB 的 `gir1.2-aravis-0.8` 即得 Python 接口 |
| B. 海康 MVS SDK | 未使用 | 官方路线，需注册下载；优势是参数最全。目前没必要 |
| C. harvesters + GenTL | 不可用 | 需要 .cti producer，aravis 不提供 |

**关键结论：之前失败不是软件不兼容，是 USB 口的问题。**
换到机身原生 USB3 口后，`arv-test-0.8` 的 `SingleAcquisition` /
`SoftwareTrigger` / `MultipleAcquisitionB` 全部 SUCCESS，本项目的
`calib/camera.py` 实测 **零丢包、帧间隔精确 10.00 ms**。

```bash
sudo apt install -y gir1.2-aravis-0.8        # 29.7 kB，装完即可用 Python 取流

# 自检与取图
python3 calib/camera.py --list                       # 列出设备
python3 calib/camera.py --source hik --info          # 打印能力与参数范围
python3 calib/camera.py --source hik --grab 5 --out /tmp/hik \
    --width 1624 --height 1240 --pixel-format Mono8 --exposure 20000 --gain 12

# 若还要用 aravis 自身工具
lsusb -t | grep -B1 -A3 2bdf      # 链路速率必须是 5000M；变成 480M 就是线/口退化
timeout 30 arv-test-0.8           # 各功能项自检
```

**这块相机最值钱的特性：支持 `DeviceTimestamp`**（1 GHz tick，实测帧间隔
精确 10.00 ms）。它不随主机负载抖动，比主机接收时刻干净得多，对 C2 时间对齐
是实质收益。`calib/camera.py` 会把设备时间戳与主机单调时钟一起给出来。

**像素格式选 Mono8。** 相机是彩色的，但实测 `Mono8` 由**相机内部**完成灰度
转换（对角差 1.26 vs 轴向差 1.07；若是把原始 Bayer 拼图直接丢出来会出现
2~3 倍的高频棋盘格伪影，会严重伤害亚像素角点检测）。用 `BayerRG8` 也可以，
但需要自己 `cv2.cvtColor(..., COLOR_BayerRG2GRAY)`。

**⚠️ 遇到"时好时坏"先怀疑线/口。** 本次就是换了一个 USB 口就全好了。
寄存器读写时好时坏 + 取流超时，典型指向 USB3 链路质量（线缆/供电/主控口），
不要在软件层死磕。

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

---

## 13. 工具与文档索引

### 13.1 源码 `calib/`（标定核心，全部纯 numpy/scipy/OpenCV）

| 文件 | 作用 |
|---|---|
| `imu_h7.py` | H7 IMU 二进制协议解析（82 字节帧 @1 kHz，含完整逆向证据链；直接运行=协议自检） |
| `camera.py` | 统一相机层：U3V(aravis) + UVC(OpenCV)，输出**设备硬件时间戳 + 主机单调时钟**双时间戳 |
| `exposure.py` | 曝光质量判定（**饱和像素占比**为主判据 + 可执行的曝光建议；直接运行=自测） |
| `target_detect.py` | 棋盘格(SB 检测器) / ChArUco 检测；`solve_pnp_board` 处理**角点顺序约定**与**平面二义性** |
| `intrinsics.py` | **C1** 内参（张正友 + Brown-Conrady），**强制 train/val 分离**验证 |
| `sync.py` | **C2** 时间对齐（转动角度匹配，**不依赖外参**）+ 零偏标定 + 陀螺积分 |
| `extrinsics.py` | **C3** 外参：手眼 AX=XB（四解法 + 一致簇投票 + 帧级异常剔除）+ 杠杆臂（重力联合估计 + Huber IRLS） |
| `trajectory.py` | **B样条轨迹**：四元数半球对齐 + 归一化四元数的解析一/二阶导（比数值微分精度高约 100 倍） |
| `validate.py` | **C4** 验证：合格判据表 + Allan 方差 + 交叉验证 + Bootstrap + Markdown 报告生成 |
| `run_calibration.py` | **主入口**：配置驱动，一条命令跑完 C1→C2→C3→C4 |
| `io_data.py` | 数据集与结果读写（YAML/JSON 可回读） |
| `simulate.py` | 数字孪生：真值已知的相机+IMU 刚体，用于合成验证 |

### 13.2 工具 `tools/`

| 工具 | 作用 |
|---|---|
| `capture_h7.py` | **数据采集**（工业相机 / UVC + H7 IMU），带**角速度**与**深度多样性**实时反馈 |
| `make_board.py` | 生成**页面尺寸精确到 0.01 mm** 的可打印标定板（附 100 mm 校验尺） |
| `compare_with_ros.py` | **与成熟工具对比**：ROS `camera_calibration` 无头标定 + **投影函数级**对比 + 半径分布 +（合成数据）**真值裁决** |
| `setup_ros_bypass.sh` | **免 sudo** 抽出 ROS `camera_calibration` 到 `third_party/roscc`（不需要 topic / GUI / 相机，缓存 .deb 可离线重建） |
| `hunt_board.py` | **不用弹窗找板子**：连拍一组后自动试各种规格，附带曝光判定 |
| `selftest_intrinsics.py` | C1 两层自检（点级验数学 / 图像级验端到端） |
| `selftest_pipeline.py` | C1+C2+C3 五项自检（含手眼约定校验、模拟器导数精度） |
| `selftest_capture.py` | 采集工具**离线整链路自测**（伪造 H7 串口 + 真实 UVC 相机） |
| `experiment_depth_diversity.py` | 受控实验：**深度跨度**对内参可辨识性的影响 |
| `make_synthetic_session.py` | 造一份**落盘**的合成数据集（含真值），用于端到端验证交付物 |
| `probe_hardware.py` / `check_motion.py` | 硬件自检 / 运动激励体检（早期工具，仍可用） |

### 13.3 文档 `doc/`

| 文档 | 内容 |
|---|---|
| **[标定板选择指南.md](doc/标定板选择指南.md)** | 板子该印多大、可用距离区间、打印 6 要点 |
| **[答辩材料.md](doc/答辩材料.md)** | 60 秒开场白、验收 8 条应答、答辩 8 问详解、**失效边界**、演示脚本、可能追问的硬问题 |
| **[对标RM主流做法.md](doc/对标RM主流做法.md)** | RM 圈通用做法（相机→云台枪管 + 编码器位姿）与我们的差异 |
| **[方法定位与主流对比.md](doc/方法定位与主流对比.md)** | 三大流派、逐环节对照、优势/劣势/缺口 |
| **[算法清单与可选高级方法.md](doc/算法清单与可选高级方法.md)** | 已用算法 + 可升级方向 + 优先级建议 |
| **[H7_IMU协议.md](doc/H7_IMU协议.md)** | 协议逆向结论与证据链 |
| **[实施计划_5天.md](doc/实施计划_5天.md)** / **[项目状态汇报_Camera-IMU标定.md](doc/项目状态汇报_Camera-IMU标定.md)** | 计划与进度汇报 |
| `相机取流诊断.md` / `IMU板排错清单.md` / `ROS2上手说明.md` | 历史排查记录 |

### 13.4 五条对采集最关键的实测结论

1. **角速度决定时间对齐精度**：`τ 分辨率 ≈ 角度残差 / 角速度`。转慢了 τ 必不准 → 目标 **≥ 90 °/s**。
2. **深度跨度决定内参可辨识性**：跨度 1.0× 时 `fx` 偏 **+4.17%**，而重投影误差只有 0.13 px
   **不会报警** → 必须走遍近/中/远，**跨度 ≥ 2×**。
3. **开头静止段是零偏标定的唯一来源**：没有它，陀螺零偏只能被（正确地）拒绝，
   未补偿的零偏会污染后面每一步。
4. **曝光错了整场白采**：过曝时白格与背景一起顶到 255，角点一个都检不出
   （2026-09-14 第一次真机试采集就是这样报废的）。判据是**饱和像素占比**，
   不是均值 —— 棋盘格黑白各半，均值天然居中，过曝帧的均值可能只是"看着正常"的 225。
5. **板子尺度错了几乎不报警**：把 20 mm 报成 40 mm，内参/畸变/`R_IC`/`τ` 逐位不变，
   `|t_IC|` 只动 7.5%。**重投影残差小 ≠ 参数对** —— 这是第 2 条和第 5 条的共同教训。
