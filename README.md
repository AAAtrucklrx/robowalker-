# Camera-IMU 联合标定

RoboWalker 算法组 2027 新生提前批大作业 · 任务 4 实现。

对机械固连的相机与 IMU，完成 **相机内参标定** 与 **Camera-IMU 外参标定**，
并给出可复现的误差验证。

> 本文档按提交要求组织为三部分：
> **① 硬件数据如何读取 → ② 具体算法实现 → ③ 算法输出什么**
>
> 详细版说明（含完整排错记录与实测数据）见 [`doc/详细实现说明.md`](doc/详细实现说明.md)。

---

# ① 硬件数据如何读取

本任务的两个传感器**都不能用常规方式读取**，这是实现中最先要解决的问题。

## 1.1 相机：海康 MV-CS020-10UC

| 项 | 值 |
|---|---|
| 型号 | Hikrobot MV-CS020-10UC |
| 接口 | **USB3 Vision（U3V）—— 不是 UVC** |
| 分辨率 | 1624×1240，Mono8 |
| 帧率 | 10 fps |
| 时间戳 | 相机内置 `DeviceTimestamp`，**1 GHz** |

**关键点：USB3 Vision 不是 UVC 协议。**

这意味着 `cv2.VideoCapture(0)` **读不到这台相机**（UVC 才走 V4L2）。
必须走厂商的 GenICam 协议栈。

实现见 [`calib/camera.py`](calib/camera.py)，做了**两层抽象**：

```
camera.py
├── HikCamera    —— 海康 U3V 工业相机，走 aravis（GenICam 协议栈）
└── UvcCamera    —— 普通 UVC 摄像头，走 OpenCV，便于没有工业相机时调试
        ↑
   统一接口 CameraBase：open() / frames() / close()
```

**为什么要有 UVC 分支**：工业相机不在手边时，整条管线仍然可以用普通摄像头跑通。

### 踩过的三个坑（都在代码里留了防护）

| 现象 | 根因 | 处理 |
|---|---|---|
| 进程随机 `SIGABRT`（`corrupted size vs. prev_size`） | `np.frombuffer` 零拷贝取图，底层 buffer 被 GC 后悬空指针（use-after-free） | 改为 `np.array(..., copy=True)` 强制拷贝 |
| 连续开关 5 次后相机"假死"，报 `LIBUSB_ERROR_BUSY` | 非正常释放导致 USB 端点未复位 | 检测到 wedge 签名后自动做 USB reset（`USBDEVFS_RESET`）|
| `read_exposure()` 报错 | 调了不存在的方法 `is_exposure_time_auto()` | 改为 `get_exposure_time_auto()` |

**实测数据（2026-09-13）**：取流已打通，稳定输出 1624×1240 Mono8。

## 1.2 IMU：H7_IMU_With_EKF

| 项 | 值 |
|---|---|
| USB VID:PID | `0483:6666` |
| 接口 | 串口（CDC） |
| 帧长 | **82 字节** |
| 同步头 | `5A A5` |
| 频率 | **1000.000 Hz** |
| 时间戳 | 帧内 `t_board_ms`（板载毫秒计数） |

**帧结构**（偏移单位：字节）见 [`doc/H7_IMU协议.md`](doc/H7_IMU协议.md)，实现见
[`calib/imu_h7.py`](calib/imu_h7.py)：

| 偏移 | 长度 | 内容 |
|---|---|---|
| 0 | 2 | 同步头 `5A A5` |
| 2 | 2 | 帧长 `0x004C` = 76 |
| 14 | 4 | `t_board_ms` 板载时间戳 |
| 18 | 12 | 加速度 x,y,z |
| 30 | 12 | 角速度 x,y,z |
| 42 | 12 | 磁力计 x,y,z（**实测恒为 0**）|
| 66 | 16 | 四元数 w,x,y,z |

### 关键实测结论

**磁力计恒为 0 → yaw 不可观测。**
这直接决定了后续算法设计：**所有用到旋转的地方只用角速度积分，绝不依赖磁力计或绝对航向。**
详见第 ② 部分 C2/C3。

**时间戳的坑**：串口重连时空闲会出现
`device reports readiness to read but returned no data`，
已在 `capture_h7.py` 里做 3 次重试。

## 1.3 两个时钟不同源 —— 这是 C2 存在的理由

相机时间戳来自**相机内部 1 GHz 计数器**，IMU 时间戳来自 **STM32 板载毫秒计数器**，
两者**没有共同时间基准**，且存在固定偏移 τ。

> 这就是 **C2 时间对齐**要估的量。它在物理上真实存在，不是可选步骤。

---

# ② 具体算法实现

## 2.0 总览：四步管线

```
C0 采集         C1 内参          C2 时间对齐       C3 外参          C4 验证
────────  →  ────────────  →  ─────────────  →  ────────────  →  ──────────
图 + IMU      fx fy cx cy       τ（时钟偏移）     R_IC, t_IC       误差报告
              畸变系数                                              + 自检
```

各步的代码位置：

| 步骤 | 文件 | 行数 |
|---|---|---|
| C1 内参 | [`calib/intrinsics.py`](calib/intrinsics.py) | 293 |
| C2 时间对齐 | [`calib/sync.py`](calib/sync.py) | 427 |
| C3 外参 | [`calib/extrinsics.py`](calib/extrinsics.py) | 605 |
| C4 验证 | [`calib/validate.py`](calib/validate.py) | 362 |
| 靶标检测（C1/C3 共用）| [`calib/target_detect.py`](calib/target_detect.py) | 473 |
| 连续轨迹（C2 用）| [`calib/trajectory.py`](calib/trajectory.py) | 305 |
| 总调度 | [`calib/run_calibration.py`](calib/run_calibration.py) | 372 |

**全部纯 numpy / scipy / OpenCV 实现，不依赖任何标定框架。**

## 2.1 C1 · 相机内参标定

**做法**：张正友标定法（Zhang 2000）+ Brown-Conrady 畸变模型，
`cv2.calibrateCamera`（Levenberg-Marquardt 非线性最小二乘，
张正友闭式解给初值）。

**关键实现细节**：

| 细节 | 为什么 |
|---|---|
| 用 `findChessboardCornersSB`（sector-based）而非经典版 | 更快、更稳；且**禁止回退到经典版**（经典版慢了 684 ms 且更差）|
| 角点做 `cornerSubPix` 亚像素精化 | 内参精度直接取决于角点定位精度 |
| 校正**角点顺序约定**与 `object_points()` 一致 | 约定不一致会让重投影从 0.13 px 恶化到 108.8 px |

## 2.2 C2 · 相机-IMU 时间对齐

**物理量**：估计两个时钟之间的常量偏移 τ（`t_cam = t_imu + τ`）。

**做法（角度匹配，与坐标系无关）**：

```
① 相机侧：把每帧的 PnP 位姿拟合成 B 样条连续轨迹，解析求导得角速度
② IMU 侧：陀螺积分得角度曲线
③ 在 τ 的搜索范围内，找让两条"角度曲线"最吻合的 τ
```

**为什么用角度而不是直接比较角速度**：
角度是角速度的积分，对高频噪声有平滑作用；且**两条角度曲线的相对平移量就是 τ**，
曲线越陡（转得越快）τ 越准。

**关键实现细节**：

| 细节 | 为什么 |
|---|---|
| B 样条轨迹 + **解析导数** | 直接对离散 PnP 位姿差分会被位姿噪声放大 |
| **10 kHz 累积陀螺积分** | 标称 1000 Hz，累积积分提高时间分辨率 |
| 静止段检测先做**中位滤波去尖峰** | 实测 IMU 每秒有一个孤立单帧尖峰 `[1.44 1.45 15.33 1.79 1.89]`，滚动标准差被撑爆 → 真静止被判成运动 → 零偏标定整个放弃。去尖峰后恢复 **6901 帧 = 6.9 s** |
| 零偏用**中位数**而非均值 | 对残余尖峰鲁棒 |

**实测**：修复后在真实数据上稳定检出 6.9 s 静止段（修复前为 0 帧）。

## 2.3 C3 · Camera-IMU 外参

**输出** `T_IC`（相机系 C → IMU 系 I）：`p_I = R_IC · p_C + t_IC`。

### 2.3.1 旋转部分

用 **手眼标定方程 AX = XB**，`cv2.calibrateHandEye` 提供四种解法
（TSAI / PARK / HORAUD / DANIILIDIS），实现里**四种都跑 + 一致性聚类投票**，
而不是只用一种。

**为什么聚类**：不同解法对噪声敏感度不同，四种结果若聚在一起说明解可信；
若散开则说明数据有问题 —— 这本身就是一个数据质量判据。

### 2.3.2 平移部分

平移比旋转难得多，因为 `t_IC` 只在**相机有旋转**时才在加速度计上可观测
（旋转木马效应：离旋转轴越远，离心/切向加速度越大）。

| 手段 | 作用 |
|---|---|
| **B 样条解析求导**得角加速度和向心加速度 | 避免数值差分放大噪声 |
| **联合估计重力方向** | 重力与杠杆臂耦合，分开估会互�相污染 |
| **Huber IRLS** | 抑制离群帧 |
| **截断式离群剔除** | 剔除残余大残差帧 |
| **Allan 方差** | 分析 IMU 噪声特性，给权重提供依据 |
| **Bootstrap** | 给出参数的置信区间 |
| **交叉验证** | 用留出集检查是否过拟合 |

## 2.4 靶标检测（C1/C3 共用）

支持三种靶标，**切换靶标不需要改算法**：

| 靶标 | 状态 | 用途 |
|---|---|---|
| **棋盘格** | ✅ 主力 | 实战场景；角点精度最高（两直线求交 + 亚像素）|
| **ChArUco** | ✅ 已实现 | 抗遮挡版棋盘格（带唯一 ID）|
| **AprilGrid** | ✅ 已实现 | 用于跑公开数据集 |

**为什么要有 ChArUco**：棋盘格有一个**根本性缺陷 —— 180° 中心对称二义性**，
平面 PnP 因此存在两解。ChArUco 每个方块有 ID，可根治。

**AprilGrid 支持是后期为验证而加的**：公开的相机-IMU 数据集（EuRoC、TUM VI、Monado）
几乎都用 AprilGrid，因为每个 tag 有唯一编号，对运动模糊和部分遮挡鲁棒得多。

> ⚠️ 实测踩坑：**OpenCV 的 `DICT_APRILTAG_36h11` 检不出 Kalibr 的 AprilGrid 靶标**
> （在完全干净的靶标渲染图上检出 **0** 个，而 OpenCV 自己生成的 36h11 标记自检 **6/6 全对**）。
> `DetectorParameters.detectInvertedMarker` 对 AprilTag 字典无效（只对 ArUco 生效）。
> 最终改用**官方 `apriltag` 库**（Kalibr 用的就是它），检出率从 0/40 帧提升到 32/40 帧。

## 2.5 配置不硬编码

所有参数（靶标尺寸、相机参数、IMU 参数、算法超参）集中在
[`config/calibration.yaml`](config/calibration.yaml)，代码里没有硬编码常量。

---

# ③ 算法输出什么

## 3.1 输出文件

一条命令生成**两种格式**（内容相同），见 [`calib/io_data.py`](calib/io_data.py)：

```
results/calibration_result.yaml    ← 人读
results/calibration_result.json    ← 程序读
```

**样例**：[`results/calibration_result.yaml`](results/calibration_result.yaml)

## 3.2 输出内容

```yaml
# ── C1 相机内参 ──────────────────────────────
intrinsics:
  camera_matrix: [fx, 0, cx,
                  0, fy, cy,
                  0,  0,  1]        # 3x3 内参矩阵
  dist_coeffs: [k1, k2, p1, p2, k3] # Brown-Conrady 畸变
  image_size: [w, h]
  rms_px: 0.22                      # 重投影残差（像素）

# ── C2 时间对齐 ──────────────────────────────
time_offset:
  tau_s: -0.0123                    # 时钟偏移（秒），t_cam = t_imu + tau

# ── C3 外参 ─────────────────────────────────
extrinsics:
  R_IC: [...]                       # 3x3 旋转：相机系 → IMU 系
  t_IC: [x, y, z]                   # 平移（米）
  T_IC: [...]                       # 4x4 齐次矩阵（R_IC 和 t_IC 的合并）

# ── C4 验证指标 ──────────────────────────────
validation:
  reproj_rms_px: ...                # 重投影残差
  cross_val_rms_px: ...             # 交叉验证残差
  motion_consistency_deg: ...       # 运动一致性（旋转）
  gravity_deviation: ...            # |g| 内建自检（应 ≈ 9.80665）
  bootstrap_ci: {...}               # Bootstrap 置信区间
```

## 3.3 每一项怎么被下游用

| 输出 | 比赛时怎么用 |
|---|---|
| `camera_matrix` + `dist_coeffs` | **像素坐标 → 相机坐标系方向**（去畸变 + 反投影）|
| `R_IC` | 相机系方向 → **IMU 系方向** |
| `t_IC` | 修正杠杆臂：相机不在 IMU 原点，旋转会带来额外位移 |
| `tau_s` | 把图像时间戳对齐到 IMU 时间轴，**否则姿态数据用错时刻** |

## 3.4 坐标系定义

**任务文档 3.2 与验收第 8 条明确要求说明此项。**

`T_IC` 表示相机系 C 到 IMU 系 I 的变换：

```
p_I = R_IC · p_C + t_IC
```

> ⚠️ **与 Kalibr 的约定相反**。Kalibr 的 `T_cam_imu` 是反方向，
> 对照时必须转置，否则会得到一个"看起来合理但完全错误"的结果。

---

# ④ 快速开始

完整说明见 [`doc/详细实现说明.md`](doc/详细实现说明.md)。

```bash
# 安装（Ubuntu 24.04，系统 Python 受 PEP 668 保护）
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# ① 硬件自检
.venv/bin/python tools/probe_hardware.py

# ② 采集
.venv/bin/python tools/capture_h7.py --auto-session data/session_01

# ③ 标定：跑完 C1→C2→C3→C4 并生成报告
.venv/bin/python -m calib.run_calibration --session data/session_01

# 不依赖硬件，验证算法链路本身
.venv/bin/python -m calib.simulate --out /tmp/sim && \
.venv/bin/python -m calib.run_calibration --session /tmp/sim

# 与成熟工具对照（ROS camera_calibration，免 sudo）
bash tools/setup_ros_bypass.sh
```

---

# ⑤ 目录结构

```
calib/          标定核心库（纯 numpy/scipy/OpenCV）
  camera.py         ① 相机抽象层（U3V + UVC）
  imu_h7.py         ① H7 IMU 串口协议解析
  target_detect.py  ② 靶标检测（棋盘格 / ChArUco / AprilGrid）
  intrinsics.py     ② C1 内参
  sync.py           ② C2 时间对齐
  extrinsics.py     ② C3 外参
  trajectory.py     ② 连续时间轨迹（B 样条）
  validate.py       ② C4 验证
  io_data.py        ③ 输入输出格式
  run_calibration.py  总调度
tools/          采集、自检、对照、基准测试工具
config/         参数配置（不硬编码）
boards/         靶标文件（可直接打印）
doc/            技术文档
results/        标定输出 + 验证证据
third_party/    ROS camera_calibration 旁路（对照用）
```

---

# ⑥ 已知限制

1. **真实相机-IMU 数据尚未采到合格样本**。识别标定板这一步已打通，
   但联合标定要求相机与 IMU **刚性固连**，手持会让两者相对位置变化、
   破坏标定前提。实测手持采集的角速度、深度跨度、检测率均不达标。
   **需要刚性支架 + 云台 + 补光。**
2. **方法是"分步几何"**，对单帧位姿精度敏感；Kalibr/Basalt 的**批量优化**
   能容忍更差的检测。实测边界见 [`doc/答辩材料.md`](doc/答辩材料.md)。
3. **磁力计恒为 0**，yaw 不可观测，标定只能依赖陀螺与加速度计。
