# config/ 目录说明

## 内置场景 yaml（全量参数，一种模式一份）

MobileNet 配置按输入方式区分：

| `params_file` | `input_msg` | 典型上游 / 用途 |
|---------------|-------------|-----------------|
| `ai_infer_mobilenet.yaml` | `tb_img` | usb_cam `mjpeg2nv12_640x480_zc.yaml` → `/tbmem_img`（在线默认） |
| `file_mobilenet.yaml` | `file` | 离线单张 JPG，`file_cli.launch.py` |
| `sensor_image_mobilenet.yaml` | `sensor_image` | usb_cam |


> MobileNet 前处理使用 `tacv_dma`。`tb_img` 输入应来自相机发布的 NV12 图像；
> `sensor_image` 输入使用 `sensor_msgs/Image` 路径。

### 与 topsbot_usb_cam 成对关系

| usb_cam `params_file` | MobileNet `params_file` | 输入话题 |
|-----------------------|-------------------------|----------|
| `mjpeg2nv12_640x480_zc.yaml` | `ai_infer_mobilenet.yaml` | `/tbmem_img` |

模型切换通过修改 yaml 中的 `model.path`，或使用 launch 的 `model_path` 参数；

## 其他文件

| 文件 | 说明 |
|------|------|
| `models_manifest.yaml` | MobileNet 模型下载清单（**不**作为节点参数加载） |
| `params_reference.yaml` | 在线节点参数字典（**不自动加载**） |
| `imagenet_labels.txt` | ImageNet 1000 类标签 |

```bash
ls $(ros2 pkg prefix topsbot_mobilenet_cls)/share/topsbot_mobilenet_cls/config/*.yaml
```

## 图像订阅参数（TOPSBOT 软性约定）

| 参数 | 说明 |
|------|------|
| `input_topic` | 订阅话题（与上游成对写在 yaml） |
| `input_msg` | 在线节点使用 `tb_img` / `sensor_image`；file 工具使用 `file` 场景 |
| `tb_img_profile` | **仅** `input_msg=tb_img`：`480p` / `540p` / `1080p` / `auto` |

## 参数合并顺序（后者覆盖前者）

1. `params_file` — 上表中的场景 yaml
2. launch 覆盖 — 在线 launch 支持 `input_topic`、`input_msg`、`tb_img_profile`、
  `classifications_topic`、`model_path`（非空才写入）
3. file launch 支持 `image_file`、`output_image_file`、`model_path`

```bash
ros2 launch topsbot_mobilenet_cls mobilenet.launch.py
ros2 launch topsbot_mobilenet_cls mobilenet.launch.py input_topic:=/cam0/tbmem_img
ros2 launch topsbot_mobilenet_cls cam_mobilenet_webviz_demo.launch.py
```

`cam_mobilenet_webviz_demo.launch.py` 默认同时启动 `topsbot_usb_cam`、MobileNet 分类节点和
`topsbot_webviz`；浏览器地址为 `http://<板卡IP>:8000/`。无界面测试：
`enable_webviz:=false`。

## 话题

| 方向 | 话题 | 说明 |
|------|------|------|
| 订阅 | yaml 中 `input_topic` | 默认 `/tbmem_img` 或 `/image_raw` |
| 发布 | `/classifications` | `tb_cls_msgs/msg/TbClassifications`，按置信度发布 Top-K |

离线 `file` 模式不发布 ROS 话题，分类结果打印到终端，并输出带 Top-K
文字的 JPG 到 `output_image_file`。

### file 场景（`file_cli.launch.py`）

**默认启动**

```bash
ros2 launch topsbot_mobilenet_cls file_cli.launch.py
```

默认：MobileNetV2（`file_mobilenet.yaml`）· 输入
`share/topsbot_mobilenet_cls/imgs/test.jpg` · 输出
`/tmp/topsbot_ai_infer/out_mobilenetv2.jpg`

**一条命令改模型 / 输入 / 输出**

```bash
ros2 launch topsbot_mobilenet_cls file_cli.launch.py \
  params_file:=file_mobilenet.yaml \
  image_file:=/path/to/in.jpg \
  output_image_file:=/path/to/out.jpg
```

| launch 参数 | 作用 | yaml 是否读取 |
|-------------|------|---------------|
| `params_file` | 选择 file 场景及默认输出 | `model.*`、`preprocess.*`、`postprocess.*`、`output_image_file` |
| `image_file` | 输入 JPG | 否（launch 专用） |
| `output_image_file` | 输出 JPG | 可被 launch 覆盖 yaml |
| `model_path` | 覆盖 MobileNet `.nb` 路径 | 否（launch 专用） |

启动后打印 `[file_cli] params_file: ...`。板端修改 yaml 后需重新构建：

```bash
colcon build --packages-select topsbot_mobilenet_cls
```
