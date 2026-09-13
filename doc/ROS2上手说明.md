# ROS 2 是什么、你手上有什么、怎么用

> 面向完全没接触过 ROS 2 的人。所有命令都在你本机实测通过。
> 环境：Ubuntu 24.04（原生，非 WSL）+ ROS 2 Jazzy + X11 图形界面

---

## 一、ROS 2 到底是什么

**一句话**：ROS 2 不是操作系统，而是**让机器人上几十个程序能互相通信的中间件 + 配套工具**。

### 它解决的问题

机器人上有：摄像头程序、IMU 程序、雷达程序、路径规划程序、电机控制程序……
这些是**不同的人写的、独立的程序**，怎么让它们互相传数据？

传统做法：自己写 socket、定协议、处理重连——又累又容易错。

ROS 2 的做法：**大家约好中间有个"公告板"**。
- 发数据的程序把数据贴到公告板上（叫**发布**）
- 要数据的程序自己去公告板取（叫**订阅**）
- 谁都不需要知道对方在哪、是什么语言写的

### 底层是什么

ROS 2 的通信建立在 **DDS**（Data Distribution Service，工业级发布-订阅协议）之上。这带来两个直接影响：

1. **自动发现**：同一网段内，节点启动后自动找到彼此，不用配置 IP
2. **会"串台"**：如果同一网段有两组机器人，它们会互相看到对方的节点！
   → 所以训练现场要给每组分配不同的 **`ROS_DOMAIN_ID`**（相当于"频道号"）

---

## 二、核心概念（只需先记住 5 个）

| 概念 | 是什么 | 类比 |
|---|---|---|
| **节点 Node** | 一个独立运行的程序 | 一个工人 |
| **话题 Topic** | 一条命名的数据通道，发布/订阅 | 一个广播频道 |
| **消息 Message** | 话题里传的数据，有严格的类型 | 广播里的内容格式 |
| **服务 Service** | 一问一答的同步调用 | 打电话问事 |
| **参数 Parameter** | 节点运行时可调的配置项 | 机器上的旋钮 |

再补充三个你会用到的：

| 概念 | 用途 |
|---|---|
| **包 Package** | ROS 2 的代码组织单位，一个包 = 一组节点/库/配置 |
| **launch 文件** | 一条命令启动一堆节点（否则要开一堆终端） |
| **工作区 Workspace** | 你自己写代码的地方（`~/ros2_ws`），编译产物放这里 |

**重要约定**：消息类型写成 `包名/msg/类型名`，例如 `sensor_msgs/msg/Imu`
就是"`sensor_msgs` 这个包里名为 `Imu` 的消息类型"。**IMU 数据在 ROS 2 里就对应这个类型**。

---

## 三、你本机的实际状态（实测）

| 项目 | 状态 |
|---|---|
| ROS 2 版本 | Jazzy（`ROS_VERSION=2`, `ROS_DISTRO=jazzy`） |
| 已安装包数 | **297 个** |
| 自动加载 | ✅ 已写入 `~/.bashrc`（新终端自动生效） |
| `ros2` CLI | ✅ `/opt/ros/jazzy/bin/ros2` |
| `colcon`（编译工具） | ✅ `/usr/bin/colcon` |
| `rosdep`（依赖管理） | ✅ 已装 |
| **RViz2**（3D 可视化） | ✅ 实测启动成功，OpenGL 4.6 |
| `rqt` / `rqt_graph` | ✅ 已装 |
| `rqt_image_view` | ✅ 已装 |
| 演示包 | ✅ `demo_nodes_cpp` / `demo_nodes_py` / `turtlesim` / `image_tools` |
| 工作区 | `~/ros2_ws/src`（已建，空的） |
| 图形环境 | ✅ X11，`DISPLAY=:1` |

**还没装的（按你的任务判断要不要装）**：

| 工具 | 用途 | 你需要吗 |
|---|---|---|
| `rqt_plot` | 实时画曲线（看 IMU 数据流） | **建议装**，调试 IMU 很有用 |
| `rqt_console` | 集中看各节点日志 | 建议装 |
| `rqt_reconfigure` | 运行时调参数 | 可选 |
| `camera_calibration` | ROS 官方相机标定工具 | **值得参考**（但你的任务要求自己实现） |
| `usb_cam` / `v4l2_camera` | 把摄像头接进 ROS 2 | 可选 |
| `image_pipeline` | 图像处理管线 | 可选 |
| Gazebo / `ros_gz` | 3D 物理仿真 | 任务 6（无人机）才需要 |

一条命令补齐常用 GUI 工具：

```bash
sudo apt install -y ros-jazzy-rqt-plot ros-jazzy-rqt-console ros-jazzy-rqt-reconfigure \
                    ros-jazzy-v4l2-camera ros-jazzy-image-pipeline
```

---

## 四、动手：7 步看懂 ROS 2（你本机实测过的流程）

### 准备：打开两个终端

每个终端都要先"加载 ROS 环境"（已写进 `~/.bashrc`，新终端自动加载，但确认一下）：

```bash
source /opt/ros/jazzy/setup.bash
```

### 第 1 步：启动一个"发布者"

在**终端 A**：

```bash
ros2 run demo_nodes_cpp talker
```

看到 `[INFO] [talker]: Publishing: 'Hello World: 1'` 就是成功了。
`ros2 run <包名> <节点名>` = 运行某个包里的某个程序。

### 第 2 步：看系统里有哪些节点

在**终端 B**：

```bash
ros2 node list
```

输出：`/talker`

**这就是 ROS 2 的第一个魔法**：`talker` 是终端 A 里的独立程序，终端 B 里另一个命令能看到它。

### 第 3 步：看有哪些"话题"（数据通道）

```bash
ros2 topic list
```

输出：

```
/chatter              ← talker 在往这里发消息
/parameter_events     ← ROS 2 自动带的
/rosout               ← ROS 2 自动带的（日志）
```

### 第 4 步：看话题的类型和内容

```bash
ros2 topic info /chatter      # 类型是什么、几个发布者/订阅者
ros2 topic echo /chatter      # 实时打印消息内容（Ctrl+C 退出）
ros2 topic hz /chatter        # 发布频率
```

`ros2 topic info /chatter` 输出：

```
Type: std_msgs/msg/String
Publisher count: 1
Subscription count: 0
```

**读法**：`std_msgs/msg/String` = `std_msgs` 包里的 `String` 消息类型，内容是字符串。

### 第 5 步：启动一个"订阅者"，看它们自动连上

在**终端 C**：

```bash
ros2 run demo_nodes_py listener
```

你会看到 `[INFO] [listener]: I heard: [Hello World: 23]` 不断出现。

**注意两件事**：
1. `talker` 是 **C++** 写的，`listener` 是 **Python** 写的——**它们照样能通信**（ROS 2 跨语言）
2. 终端 B 里再跑 `ros2 topic info /chatter`，`Subscription count` 从 `0` 变成了 `1`——**自动发现**

### 第 6 步：看"谁在跟谁说话"

```bash
ros2 node info /talker
```

输出会列出这个节点**发布什么、订阅什么、提供什么服务、有哪些参数**。

图形化版本（更直观）：

```bash
rqt_graph
```

会弹出一个窗口，画出节点和话题的连接图——**推荐第一次就打开看，比文字直观得多**。

### 第 7 步：玩个可视化例子（turtlesim）

这是 ROS 2 的经典教学程序——一只小乌龟：

```bash
# 终端 A：启动乌龟模拟器和它的控制界面
ros2 run turtlesim turtlesim_node
ros2 run turtlesim turtle_teleop_key    # 用方向键控制乌龟
```

一边用方向键开乌龟，一边在终端 B 里：

```bash
ros2 topic list                    # 看多了哪些话题
ros2 topic echo /turtle1/pose      # 实时打印乌龟的位置坐标
rqt_graph                          # 看控制链路：teleop_key → /turtle1/cmd_vel → turtlesim
```

**这个例子把 ROS 2 的骨架全展示出来了**：一个节点收键盘发布速度指令，另一个节点订阅指令并模拟运动。

---

## 五、图形化工具：都是干什么的

| 工具 | 启动命令 | 用途 | 和你的标定任务的关系 |
|---|---|---|---|
| **RViz2** | `rviz2` | **3D 可视化**：显示点云、图像、坐标系、轨迹、IMU 姿态 | ⭐ **最重要**。可以把相机画面、IMU 姿态、坐标系变换叠在一个 3D 场景里看——**验证外参对不对，一眼就能看出来** |
| **rqt_graph** | `rqt_graph` | 画节点/话题连接图 | 理解系统结构 |
| **rqt_plot** | `rqt_plot` | **实时曲线**（需装） | ⭐ 看 IMU 六轴数据流是否正常、有没有噪声 |
| **rqt_image_view** | `rqt_image_view` | 显示图像话题 | 看相机画面 |
| **rqt_console** | `rqt_console` | 集中看所有节点日志 | 排错 |
| **rqt_reconfigure** | `rqt_reconfigure` | 运行时调参数 | 调曝光/阈值等 |

### RViz2 关键概念（值得先理解）

RViz2 是"**可视化面板**"，本身**不产生数据**，只把数据画出来。核心是 **Display（显示项）**，每个显示项绑定到一个话题：

| Display 类型 | 绑定的数据 | 你会用来 |
|---|---|---|
| `Image` | `sensor_msgs/msg/Image` | 看相机画面 |
| `Imu` | `sensor_msgs/msg/Imu` | 看 IMU 姿态和加速度箭头 |
| `TF` | `/tf` 话题 | **看坐标系之间的相对变换** ← 标定结果验证的核心 |
| `Path` | `nav_msgs/msg/Path` | 看运动轨迹 |
| `PointCloud2` | 点云 | （你的任务用不到） |

**固定坐标系（Fixed Frame）**：RViz2 顶部有个 `Fixed Frame` 设置，决定"以谁为原点显示一切"。标定时你会设成 `camera` 或 `imu`，然后把另一个坐标系加进 TF 显示项——**如果外参标对了，两个坐标系在 3D 视图里的相对姿态应当与实物一致**。

---

## 六、标定任务什么时候需要 ROS 2

**直接回答：任务 4 的标定程序本身不需要 ROS 2。** 任务是纯离线数据处理：读图像 + IMU → 算内参和外参。

但 ROS 2 在三处**有实际价值**：

| 场景 | 用 ROS 2 做什么 | 价值 |
|---|---|---|
| 数据采集 | 写节点：相机 → `/camera/image_raw`，IMU → `/imu/data`，然后用 `ros2 bag record` 录包 | **时间戳统一、可靠、可回放**（比手写 CSV 规范得多） |
| 结果验证 | 把标定出的外参发成 `/tf`，在 RViz2 里看相机与 IMU 坐标系是否对齐实物 | **最直观的验证方式**，答辩时也好演示 |
| 参考实现 | 看 `camera_calibration` 包（ROS 官方的相机标定）怎么做的 | 学习标准做法 |

**下一步建议**：如果你想让交付物更专业（也更容易验收），可以走这条路：

```
相机节点 ──→ /camera/image_raw ──┐
                                 ├──→ ros2 bag record ──→ 一个 rosbag 文件（含全部数据和时间戳）
IMU 节点 ───→ /imu/data ─────────┘
```

一个 rosbag 文件就包含了标定需要的全部输入，而且**自带时间戳**——这正好满足任务文档第 2 节"数据必须具有时间戳"和第 8 节"应尽量保证两种传感器数据具有可靠的时间对应关系"。

---

## 七、最容易踩的坑

| 坑 | 现象 | 解法 |
|---|---|---|
| 忘了 `source` | `ros2: command not found` | `source /opt/ros/jazzy/setup.bash`（已写进 bashrc，通常自动） |
| **`ROS_DOMAIN_ID` 撞车** | 训练现场看到别的组的节点、数据串台 | `export ROS_DOMAIN_ID=27`（**同一组内必须一致**） |
| 多终端环境不一致 | 一个终端能跑另一个报错 | 每个终端都确认 `echo $ROS_DISTRO` 输出 `jazzy` |
| 工作区 overlay 顺序错 | 用了旧版本的包 | 先 `source /opt/ros/jazzy/setup.bash`（underlay），再 `source ~/ros2_ws/install/setup.bash`（overlay） |
| 把工作区写进 bashrc 太早 | 编译失败后每个新终端都报错 | 调试期间**手动 source**，稳定后再写 bashrc |
| 用 `sudo ros2` | 权限/环境混乱 | **不要**用 sudo 跑 ros2 |

---

## 八、你自己的工作区：`~/ros2_ws`

已建好（`~/ros2_ws/src` 目前是空的）。标准流程：

```bash
# 1. 把代码包放进 src/
cd ~/ros2_ws/src
# （你的包目录放这里）

# 2. 装依赖（第一次或换分支后）
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y

# 3. 编译
colcon build --symlink-install

# 4. 加载（每个新终端都要）
source ~/ros2_ws/install/setup.bash

# 5. 只重编某个包（快）
colcon build --symlink-install --packages-select <包名>
```

**空工作区也能编译成功**（生成 `build/` `install/` `log/` 三个目录），可以先跑一遍熟悉流程。

---

## 九、速查表

```bash
# 环境
source /opt/ros/jazzy/setup.bash
echo $ROS_DISTRO ; echo $ROS_DOMAIN_ID

# 查看系统现状
ros2 node list                  # 有哪些节点
ros2 topic list                 # 有哪些话题
ros2 topic info <话题>          # 话题类型、发布/订阅者数量
ros2 topic echo <话题>          # 实时打印内容
ros2 topic hz <话题>            # 发布频率
ros2 node info <节点>           # 节点的发布/订阅/服务/参数
ros2 pkg list                   # 有哪些包
ros2 interface show <类型>      # 某个消息类型的字段定义
ros2 doctor                     # 体检

# 图形化
rviz2                           # 3D 可视化
rqt_graph                       # 节点连接图
rqt_plot                        # 实时曲线
rqt_image_view                  # 图像查看

# 运行与录制
ros2 run <包> <节点>            # 运行节点
ros2 launch <包> <launch文件>   # 批量启动
ros2 bag record -a              # 录下所有话题
ros2 bag info <bag文件>         # 看 bag 里有什么
ros2 bag play <bag文件>         # 回放

# 工作区
colcon build --symlink-install
source install/setup.bash
```
