# adoramax_realman_rm65control

Realman RM65 单臂 XR 遥操数据采集代码：**左臂**与**右臂**各一套，基于 XRoboToolkit（XR 手柄/头显）与 LeRobot 0.3.3 数据集格式。

> 本仓库只包含**单臂采集**（左臂 / 右臂）。双臂采集（`collect_dual_rm65_lerobot.py`）不在本仓库内。
>
> 代码导出源：`XR_Robot/XRoboToolkit-Teleop-Sample-Python`，分支 `Realman-RM65`，提交 `6d47caf`，导出日期 2026-11-09。

---

## 1. 功能概览

每套采集程序做三件事：

1. 通过 XRoboToolkit 读取 XR 手柄位姿，实时映射为 RM65 六轴关节角 + 夹爪开合（右臂用右手柄，左臂用左手柄）；
2. 同步采集两路 UVC 相机：顶部相机 + 腕部相机；
3. 按 LeRobot 0.3.3 数据集格式落盘（parquet + mp4），支持断点恢复与质量校验。

| | 左臂 | 右臂 |
|---|---|---|
| 入口脚本 | `run_left_rm65_collection.sh` | `run_right_rm65_collection.sh` |
| 采集主体 | `scripts/hardware/collect_left_rm65_lerobot.py` | `scripts/hardware/collect_right_rm65_lerobot.py` |
| 机械臂 IP | `192.168.1.18:8080` | `192.168.1.19:8080` |
| 腕部相机序列号 | `CP1Z842000C3` | `CP1Z842000W8` |
| 数据维度 | `left_q1..left_q6, left_gripper` | `right_q1..right_q6, right_gripper` |
| 图像键 | `observation.images.top` / `observation.images.left_wrist` | `observation.images.top` / `observation.images.right_wrist` |
| 纯遥操测试模式 | 无 | 有（`--teleop-only`） |

顶部相机两支共用同一台：序列号 `CP1Z842000EP`。

---

## 2. 目录结构

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── run_left_rm65_collection.sh          # 左臂启动脚本（含环境校验、端口占用检查、supervisor）
├── run_right_rm65_collection.sh         # 右臂启动脚本
├── scripts/
│   └── hardware/
│       ├── collect_left_rm65_lerobot.py     # 左臂采集主体
│       ├── collect_right_rm65_lerobot.py    # 右臂采集主体（含 --teleop-only）
│       └── supervise_dual_rm65_collection.py # 采集进程守护（左右臂共用）
└── xrobotoolkit_teleop/
    ├── collection/                      # 采集框架：runtime / recovery / episode / keyboard
    │                                    # cameras / dataset / preview / vr_video / supervisor
    ├── hardware/
    │   ├── right_rm65_collection_controller.py          # 右臂采集控制器
    │   └── commissioned_left_rm65_collection_controller.py  # 左臂采集控制器
    └── vendor/
        ├── xr_robot1_right_rm65/        # 冻结版 RM65 遥操内核（IK、跟随、限速、日志）
        │   ├── config/rm65_192_168_1_19.json
        │   └── rm65_teleop/
        └── xr_robot1_dual_rm65/config/rm65_192_168_1_18.json  # 左臂安装参数（数据文件，供左控制器读取）
```

---

## 3. 环境要求

- Ubuntu 22.04（推荐，与实机一致；22.04 以外未验证）
- Python 3.10（实机为 3.10.20）
- 可用的 conda 环境（实机环境名 `xr-rm65-collection`）
- XRoboToolkit PC Service Pybind（提供 `libPXREARobotSDK.so`）
- Realman RM65 机械臂 + 夹爪，可达 `192.168.1.18/19:8080`
- 两路 UVC 相机（顶部 + 腕部）

### 3.1 Python 依赖

```bash
conda create -n xr-rm65-collection python=3.10 -y
conda activate xr-rm65-collection
pip install -r requirements.txt
```

> 相机预览窗口需要 GUI 版 OpenCV（`opencv-python`）；若只装 `opencv-python-headless`，请加 `--no-camera-preview`。

### 3.2 XR SDK（必须）

采集靠 SDK 读取手柄数据，SDK 不在本仓库内：

```bash
mkdir -p dependencies
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git \
    dependencies/XRoboToolkit-PC-Service-Pybind
cd dependencies/XRoboToolkit-PC-Service-Pybind
bash setup_ubuntu.sh
```

启动脚本会自动在以下位置找 `libPXREARobotSDK.so`：

1. `$XR_SDK_LIBRARY_DIR`
2. `./dependencies/XRoboToolkit-PC-Service-Pybind/lib`
3. `~/XRoboToolkit-Teleop-Sample-Python/dependencies/XRoboToolkit-PC-Service-Pybind/lib`
4. `~/XRoboToolkit-Teleop-Sample-Python2/dependencies/XRoboToolkit-PC-Service-Pybind/lib`

已装好 SDK 的机器可直接指定：

```bash
export XR_SDK_LIBRARY_DIR=/path/to/XRoboToolkit-PC-Service-Pybind/lib
```

### 3.3 修改 Python 路径

两个 `run_*_collection.sh` 里写死了：

```bash
PYTHON_BIN="/home/dora/anaconda3/envs/xr-rm65-collection/bin/python"
```

换机器时改成你自己的解释器路径（或建同名环境）。

---

## 4. 运行

### 4.1 左臂采集

```bash
cd adoramax_realman_rm65control
./run_left_rm65_collection.sh \
  --dataset-root ~/datasets/left_demo \
  --repo-id local/left_demo \
  --task "把黄色积木放到左边盒子里"
```

### 4.2 右臂采集

```bash
./run_right_rm65_collection.sh \
  --dataset-root ~/datasets/right_demo \
  --repo-id local/right_demo \
  --task "把红色易拉罐放到右边盒子里"
```

### 4.3 右臂纯遥操测试（不接相机、不录数据）

```bash
./run_right_rm65_collection.sh --teleop-only --log-dir ~/outputs/teleop_test
```

### 4.4 键盘操作（录制时）

| 按键 | 作用 |
|---|---|
| `Enter` | 开始录制一个 episode |
| `→` | 结束并**保存**当前 episode |
| `←` | 结束并**丢弃**当前 episode |
| `Esc` | 退出采集 |

### 4.5 常用参数

| 参数 | 说明 |
|---|---|
| `--dataset-root` | 必填，数据集根目录 |
| `--repo-id` | 必填，LeRobot repo id，如 `local/left_demo` |
| `--task` | 必填，任务描述（写进数据集） |
| `--top-camera` | 顶部相机，默认 `auto`，可填 `/dev/videoX` 或序号 |
| `--left-wrist-camera` / `--right-wrist-camera` | 腕部相机，默认 `auto` |
| `--camera-width` / `--camera-height` / `--camera-fps` | 默认 640×480@30 |
| `--no-camera-preview` | 关闭本地预览窗口 |
| `--vr-preview` / `--no-vr-preview` | 是否向 XRoboToolkit Remote Vision 推送头显画面，默认开启 |
| `--arm-response` | 平移响应曲线：`original` / `faster` / `responsive`（默认） |
| `--forward-reach-gain` | 前向可达增益，默认 2.0 |
| `--num-episodes` | 采够 N 条自动结束 |
| `--video-crf` | AV1 编码质量，默认 18，越小越清晰 |

左臂与右臂参数基本一致；右臂额外支持 `--teleop-only`、`--log-dir`。

---

## 5. 数据格式

- 频率：30 fps
- `observation.state` / `action`：7 维（左臂 `left_q1..left_q6,left_gripper`；右臂 `right_q1..right_q6,right_gripper`）
- 图像：`observation.images.top`、`observation.images.left_wrist` 或 `observation.images.right_wrist`
- 每个 episode 一个 parquet + 两路 mp4，附 `meta/info.json`、`meta/episodes.jsonl`、`meta/tasks.jsonl`
- 目录形如 `dataset-root/collection-YYYYmmdd_HHMMSS_xxxxxx/`
- 异常中断的 episode 会进入 `recovery/`，下次启动可恢复或作废
- 落盘策略：`discard_invalid_on_save`（校验不通过的 episode 不写正式数据集）

---

## 6. 实现要点（便于二次开发）

- `collect_left_rm65_lerobot.py`
  - `LEFT_DIMENSIONS`（第 18–25 行）
  - 相机序列号常量（第 26–29 行）
  - `_run_real()`（第 273 行）真实采集主流程
  - `main()`（第 467 行）
- `collect_right_rm65_lerobot.py`
  - `RIGHT_DIMENSIONS`（第 15 行）
  - `RIGHT_WRIST_CAMERA_SERIAL`（第 17 行）
  - `--teleop-only` 参数（第 23 行）与 `_run_teleop_only` 纯遥操流程
  - 相机解析带重试：`_resolve_camera_with_retry`
- `xrobotoolkit_teleop/collection/runtime.py`：`DualRM65CollectionRuntime`，采集主循环（左右臂共用，单臂也走它）
- `xrobotoolkit_teleop/collection/recovery.py`：`RecoverableLeRobotDatasetSink`，断点恢复
- `xrobotoolkit_teleop/collection/supervisor.py`：`run_supervised`，心跳守护
- 左臂速度系数：`LEFT_SPEED_SCALE = 0.85`（左控制器）
- 左臂复用右臂标定，在 `load_left_shared_config()` 里按安装位姿做镜像变换

---

## 7. 注意事项

1. **三种模式互斥**：左臂采集、右臂采集、双臂采集都要独占 `192.168.1.18/19:8080`，启动脚本会检查端口占用；不要同时跑。
2. **启动前确认机械臂周围无人**，XR 断连时有 `xr_stale_timeout` 保护，但不应依赖。
3. 采集程序不会自动复位机械臂，录制每个 episode 前自行摆好初始位。
4. 左臂脚本没有 `--teleop-only`，纯测左臂遥操请用遥操主程序（`dual_main.py` / `main.py`），不在本仓库。
5. `--repo-id` 建议用 `local/xxx`，避免误传到 Hugging Face Hub。
6. 首次运行前建议先用 `--num-episodes 1` 采一条验证相机与机械臂映射。

---

## 8. 未包含内容

- 双臂采集（`collect_dual_rm65_lerobot.py`）与双臂遥操主程序
- `dependencies/`（XR SDK、R5、IK 依赖，体积过大，按第 3.2 节自行安装）
- `assets/`（URDF/仿真资源，采集不需要）
- 数据集本身

---

## 9. 许可

MIT，见 [LICENSE](LICENSE)。
