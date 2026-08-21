
# topsbot-mobilenetv2

[English](README.md)

# 功能介绍

`topsbot-mobilenetv2` 在 MES20 开发板上做 mobilenetv2 图像分类，支持 **MobileNetV2** 。

- **接相机**：实时识别画面中的物体，输出分类结果
- **单张图片**：对本地 JPG 做离线分类，生成叠加 Top-5 的图片
- **浏览器预览**：配合 [`topsbot_webviz`](../topsbot_webviz) 在网页上查看画面与分类结果

# 物料清单

| 序号 | 名称 | 说明 |
| ---- | ---- | ---- |
| 1 | TOPSBOT MES20 板卡 | NPU（`ta_runtime`）与 dmabuf 环境 |
| 2 | USB 摄像头 | 联合测试时配合 `topsbot_usb_cam`，建议 MJPEG 640×480 |
| 3 | MobileNetV2 模型 | `mobilenetv2_int8.nb`、`mobilenetv2_float16.nb` |
| 4 | 标签文件 | ImageNet 1000 类标签 `imagenet_labels.txt`，支持自定义 `labels_path` |
| 5 | `topsbot_webviz` | 联合测试时的 Web 预览（可选） |

# 编译与安装

## PC 端（交叉编译）

在 `topsbot_cc` 工具链环境中编译，生成供 MES20 安装的 `install/`：

```bash
# PC：topsbot_cc 工具链环境
cd topsbot_cc/topsbot_ws
./tools/build.sh -s 'tb_cls_msgs topsbot_cv topsbot_ai_infer topsbot_mobilenet_cls topsbot_webviz'
```

将 `install/` 同步到板卡，或直接在板端工作空间中编译（若板端具备完整编译环境）。

## MES20 板端（安装与加载）

**方式 A：DEB 包（推荐，源可用时）**

```bash
# MES20
sudo apt update
sudo apt install -y tbros-humble-topsbot-ros
```

**方式 B：加载工作空间**

```bash
# MES20
source /opt/ros/humble/setup.bash
#source <install>/setup.bash
```

## 模型文件

模型（`.nb`）不随源码分发。首次在板端启动时，launch 会检查 `~/topsbot/models/`，缺失则按 `config/models_manifest.yaml` 自动下载 `models.zip` 并解压。

| 模型 | 默认路径 |
| ---- | -------- |
| MobileNetV2 | `~/topsbot/models/mobilenetv2_int8.nb` |
| MobileNetV2 | `~/topsbot/models/mobilenetv2_float16.nb` |


离线环境或已手动放置模型时：

```bash
# MES20
ros2 launch topsbot_mobilenet_cls file_cli.launch.py auto_download_models:=false
```

手动下载脚本（维护者需先配置 `models_manifest.yaml` 中的 `models_zip_url`）：

```bash
# PC 或 MES20（源码树）
./scripts/download_models.sh
```

# 使用方式

场景配置为全量 yaml，位于 `config/`。命名与成对关系见 [`config/README.md`](config/README.md)。快速命令见 [`QUICKSTART.md`](QUICKSTART.md)。

## file 模式（离线单张 JPG）

适用：不接相机，快速验证模型与分类结果。Launch：`file_cli.launch.py`。

**默认启动**

```bash
# MES20
ros2 launch topsbot_mobilenet_cls file_cli.launch.py
```

| 项 | 默认值 |
|----|--------|
| 模型 | Mobilenetv2（`file_mobilenet.yaml`） |
| 输入 | `share/topsbot_yolo_det/imgs/test.jpg` |
| 输出 | `/tmp/topsbot_ai_infer/out_mobilenetv2.jpg` |

**一条命令改输入 / 输出**（只写需要改的即可）
离线单张 JPG，
```bash
# MES20
ros2 launch topsbot_mobilenet_cls file_cli.launch.py \
  image_file:=/path/to/img.jpg \
  output_image_file:=/tmp/mobilenet_top5.jpg
```

| 项 | 默认值 |
|----|--------|
| 模型 | MobileNetV2（`file_mobilenet.yaml`） |
| 输入 | `/path/to/img.jpg`|
| 输出 | `/tmp/mobilenet_top5.jpg` |

## 单节点模式（在线推理）

适用：上游已发布图像话题（如另一路 `topsbot_usb_cam` 或自定义节点）。Launch：`mobilenet.launch.py`。

**默认启动**（订阅 `/tbmem_img`，YOLO11，发布 `/classifications`）

```bash
# MES20
ros2 launch topsbot_mobilenet_cls mobilenet.launch.py
```

**常用调整**

```bash
# MES20 — 多相机 / 换订阅话题
ros2 launch topsbot_mobilenet_cls mobilenet.launch.py input_topic:=/cam0/tbmem_img

# MES20 — 跳帧降负载
ros2 launch topsbot_mobilenet_cls mobilenet.launch.py frame_skip:=2

# MES20 — sensor_msgs 路径（须与 usb_cam 成对切换）
ros2 launch topsbot_usb_cam usb_cam.launch.py params_file:=mjpeg2nv12_640x480_sensor.yaml setup_shm:=false
ros2 launch topsbot_mobilenet_cls mobilenet.launch.py params_file:=sensor_image_mobilenet.yaml
```

| launch 参数 | 作用 |
|-------------|------|
| `params_file` | 场景 yaml（模型、`input_msg`、话题、阈值等） |
| `input_topic` | 覆盖订阅话题 |
| `input_msg` | 覆盖输入类型：`tb_img` / `sensor_image` / `file` |
| `frame_skip` | 跳帧，0 表示不跳 |

查看全部 launch 参数：`ros2 launch topsbot_mobilenet_cls mobilenet.launch.py -s`

## 联合测试模式（相机 + MobileNet + Web 预览）

适用：USB 相机端到端联调。Launch：`cam_mobilenet_webviz_demo.launch.py`（一次启动 `topsbot_usb_cam`、`topsbot-mobilenetv2`、`topsbot_webviz`）。

**默认启动**

```bash
# MES20
source /opt/tbros/humble/setup.bash
#source <install>/setup.bash
ros2 launch topsbot_mobilenet_cls cam_mobilenet_webviz_demo.launch.py
```

浏览器访问 `http://<板卡IP>:8000/` 查看图像上有 Top-5 的分类结果。

默认配置：`usb_cam` `mjpeg2nv12_640x480_zc.yaml` + mobilenet `ai_infer_mobilenet.yaml` + webviz `tb_img_mobilenet.yaml`。

**常用调整**

```bash
# MES20 — 仅推理，不要 Web 预览
ros2 launch topsbot_mobilenet_cls cam_mobilenet_webviz_demo.launch.py enable_webviz:=false
```

**确认链路正常**

```bash
# MES20
ros2 topic hz /tbmem_img
ros2 topic hz /classifications
ros2 topic echo /classifications --once
```

启动成功时，mobilenet 节点日志类似：

```text
ready: backend=mobilenet model=mobilenetv2 input=tb_img:/tbmem_img workers=1 strategy=least_loaded queue=2 drop=drop_oldest classifications=/classifications
AI worker core=.. healthy=.. in_flight=.. assigned=.. completed=.. failed=.. dropped=.. latency_ewma=..ms
AI totals received=.. published=.. rejected=..
```

# 接口说明

## ROS 话题

| 方向 | 话题 | 消息类型 | 说明 |
|------|------|----------|------|
| 订阅 | `/tbmem_img`（默认） | `tb_img_msgs/TbMsg480P` 等 | `input_msg: tb_img` |
| 订阅 | `/image_raw` | `sensor_msgs/Image` | `input_msg: sensor_image` |
| 发布 | `/classifications`（默认） | `tb_cls_msgs/msg/TbClassifications` | 按置信度降序发布 Top-K 分类结果 |

`file` 模式不发布 ROS 话题，分类结果打印到终端，并将带 Top-K
文字的 JPG 写入 `output_image_file`。

## 节点参数

完整参数见
[`config/params_reference.yaml`](config/params_reference.yaml)。

| 分组 | 主要参数 | 说明 |
|------|----------|------|
| 输入 | `input_msg`、`input_topic`、`tb_img_profile` | 支持 `tb_img` 和 `sensor_image` |
| 输出 | `classifications_topic`、`frame_skip`、`frame_id` | 分类话题、跳帧和帧 ID |
| 模型 | `model.backend`、`model.name`、`model.type`、`model.path` | MobileNetV2 模型及 `.nb` 路径 |
| 前处理 | `preprocess.engine`、`input_width`、`input_height` | taCV/DMA 前处理及 `224x224` 输入尺寸 |
| 后处理 | `postprocess.top_k`、`labels_path` | 输出类别数量和标签文件 |
| 调试 | `enable_profile` | 打印 input copy、taCV、tensorize、infer、post 耗时 |

场景 yaml 为上述参数的全量实例，一种用法一份文件：

| `params_file` | `input_msg` | 模型 |
|---------------|-------------|------|
| `ai_infer_mobilenet.yaml` | `tb_img` | Mobilenetv2（在线默认） |
| `sensor_image_mobilenet.yaml` | `sensor_image` | Mobilenetv2 |
| `file_mobilenet.yaml` | `file` | Mobilenetv2（离线默认） |

参数合并顺序：`params_file` yaml → launch 覆盖项（`input_topic` `input_msg` 等，非空才生效）。详见 `config/README.md`。



## Launch 文件

| Launch | 用途 |
|--------|------|
| `file_cli.launch.py` | 离线 file 模式 |
| `mobilenet.launch.py` | 在线单节点 |
| `cam_mobilenet_webviz_demo.launch.py` | 相机 + mobilenet + Web 联合测试 |

各 launch 参数：`ros2 launch topsbot_mobilenet_cls <launch名> -s`

## 许可

Apache License 2.0 — 见 [LICENSE](./LICENSE) 与 [NOTICE](./NOTICE)。

欢迎贡献，请参阅 [CONTRIBUTING.md](./CONTRIBUTING.md)。


