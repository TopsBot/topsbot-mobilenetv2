# topsbot-mobilenetv2

[中文文档](README_cn.md)

# Overview

`topsbot-mobilenetv2` runs MobileNetV2 image classification on the MES20 board, with **MobileNetV2** support.

- **With a camera**: real-time object recognition; classification results are published for other nodes or Web UIs
- **Single image**: offline classification on a local JPG; outputs an image overlaid with Top-5 results
- **Browser preview**: use with [`topsbot_webviz`](../topsbot_webviz) to view the feed and classification results in a browser

# Bill of Materials

| # | Item | Notes |
| - | ---- | ----- |
| 1 | TOPSBOT MES20 board | NPU (`ta_runtime`) and dmabuf support |
| 2 | USB camera | For joint testing with `topsbot_usb_cam`; MJPEG 640×480 recommended |
| 3 | MobileNetV2 models | `mobilenetv2_int8.nb`, `mobilenetv2_float16.nb` |
| 4 | Label file | ImageNet 1000-class labels `imagenet_labels.txt`; custom `labels_path` supported |
| 5 | `topsbot_webviz` | Web preview for joint testing (optional) |

# Build and Install

## PC (cross-compile)

Build inside the `topsbot_cc` toolchain environment to produce `install/` for the MES20:

```bash
# PC: topsbot_cc toolchain
cd topsbot_cc/topsbot_ws
./tools/scripts/build.sh -s 'tb_cls_msgs topsbot_cv topsbot_ai_infer topsbot_mobilenet_cls topsbot_webviz'
```

Sync `install/` to the board, or build directly on the board if a full build environment is available.

## MES20 board (install and source)

**Option A: DEB packages (recommended when available)**

```bash
# MES20
sudo apt update
sudo apt install -y tbros-humble-topsbot-ros
```

**Option B: workspace overlay**

```bash
# MES20
source /opt/ros/humble/setup.bash
#source <install>/setup.bash
```

## Model files

Models (`.nb`) are not shipped with source. On first launch on the board, `~/topsbot/models/` is checked; missing files trigger an automatic download of `models.zip` per `config/models_manifest.yaml`, then extraction.

| Model | Default path |
| ----- | ------------ |
| MobileNetV2 | `~/topsbot/models/mobilenetv2_int8.nb` |
| MobileNetV2 | `~/topsbot/models/mobilenetv2_float16.nb` |

Offline or pre-installed models:

```bash
# MES20
ros2 launch topsbot_mobilenet_cls file_cli.launch.py auto_download_models:=false
```

Manual download (maintainers must set `models_zip_url` in `models_manifest.yaml`):

```bash
# PC or MES20 (source tree)
./scripts/download_models.sh
```

# Usage

Scene configs are full yaml files under `config/`. Naming and paired configs are documented in [`config/README.md`](config/README.md). Quick commands: [`QUICKSTART.md`](QUICKSTART.md).

## File mode (offline single JPG)

Use when you need a quick model check without a camera. Launch: `file_cli.launch.py`.

**Default launch**

```bash
# MES20
ros2 launch topsbot_mobilenet_cls file_cli.launch.py
```

| Item | Default |
|------|---------|
| Model | Mobilenetv2 (`file_mobilenet.yaml`) |
| Input | `share/topsbot_yolo_det/imgs/test.jpg` |
| Output | `/tmp/topsbot_ai_infer/out_mobilenetv2.jpg` |

**One command for input / output** (include only what you need)

```bash
# MES20
ros2 launch topsbot_mobilenet_cls file_cli.launch.py \
  image_file:=/path/to/img.jpg \
  output_image_file:=/tmp/mobilenet_top5.jpg
```

| Launch argument | Purpose |
|-----------------|---------|
| `params_file` | Model: `file_mobilenet.yaml`  |
| `image_file` | Input JPG path |
| `output_image_file` | Output JPG path |

## Single-node mode (online inference)

Use when an upstream node already publishes images (e.g. another `topsbot_usb_cam` or a custom node). Launch: `mobilenet.launch.py`.

**Default launch** (subscribes `/tbmem_img`, YOLO11, publishes `/classifications`)

```bash
# MES20
ros2 launch topsbot_mobilenet_cls mobilenet.launch.py
```

**Common adjustments**

```bash
# MES20 — multi-camera / different topic
ros2 launch topsbot_mobilenet_cls mobilenet.launch.py input_topic:=/cam0/tbmem_img

# MES20 — frame skip
ros2 launch topsbot_mobilenet_cls mobilenet.launch.py frame_skip:=2

# MES20 — sensor_msgs path (pair with usb_cam)
ros2 launch topsbot_usb_cam usb_cam.launch.py params_file:=mjpeg2nv12_640x480_sensor.yaml setup_shm:=false
ros2 launch topsbot_mobilenet_cls mobilenet.launch.py params_file:=sensor_image_mobilenet.yaml
```

| Launch argument | Purpose |
|-----------------|---------|
| `params_file` | Scene yaml (model, `input_msg`, topics, thresholds, …) |
| `input_topic` | Override subscription topic |
| `input_msg` | Override input type: `tb_img` / `sensor_image` / `file` |
| `frame_skip` | Frame skip; `0` means no skip |

List all launch arguments: `ros2 launch topsbot_mobilenet_cls mobilenet.launch.py -s`

## Joint test mode (camera + MobileNet + Web preview)

End-to-end USB camera test. Launch: `cam_mobilenet_webviz_demo.launch.py` (starts `topsbot_usb_cam`, `topsbot-mobilenetv2`, and `topsbot_webviz`).

**Default launch**

```bash
# MES20
source /opt/tbros/humble/setup.bash
#source <install>/setup.bash
ros2 launch topsbot_mobilenet_cls cam_mobilenet_webviz_demo.launch.py
```

Open `http://<board-ip>:8000/` in a browser to view the NV12 feed with Top-5 classification results overlaid.

Defaults: `usb_cam` `mjpeg2nv12_640x480_zc.yaml` + mobilenet `ai_infer_mobilenet.yaml` + webviz `tb_img_mobilenet.yaml`.

**Common adjustments**

```bash
# MES20 — headless (no webviz)
ros2 launch topsbot_mobilenet_cls cam_mobilenet_webviz_demo.launch.py enable_webviz:=false
```

**Verify the pipeline**

```bash
# MES20
ros2 topic hz /tbmem_img
ros2 topic hz /classifications
ros2 topic echo /classifications --once
```

On success, the mobilenet node logs look like:

```text
ready: backend=mobilenet model=mobilenetv2 input=tb_img:/tbmem_img workers=1 strategy=least_loaded queue=2 drop=drop_oldest classifications=/classifications
AI worker core=.. healthy=.. in_flight=.. assigned=.. completed=.. failed=.. dropped=.. latency_ewma=..ms
AI totals received=.. published=.. rejected=..
```

# Interface Reference

## ROS topics

| Direction | Topic | Message type | Notes |
|-----------|-------|--------------|-------|
| Subscribe | `/tbmem_img` (default) | `tb_img_msgs/TbMsg480P` etc. | `input_msg: tb_img` |
| Subscribe | `/image_raw` | `sensor_msgs/Image` | `input_msg: sensor_image` |
| Publish | `/classifications` (default) | `tb_cls_msgs/msg/TbClassifications` | Top-K classifications in descending confidence order |

File mode writes JPG to `output_image_file` instead of publishing ROS topics.

## Node parameters

Full parameters are documented in [`config/params_reference.yaml`](config/params_reference.yaml). Grouped summary:

| Group | Key parameters | Notes |
|-------|----------------|-------|
| Input | `input_msg`, `input_topic`, `tb_img_profile` | Supports `tb_img` and `sensor_image` |
| Output | `classifications_topic`, `frame_skip`, `frame_id` | Classification topic, frame skipping, and frame ID |
| Model | `model.backend`, `model.name`, `model.type`, `model.path` | MobileNetV2 model and `.nb` path |
| Preprocess | `preprocess.engine`, `input_width`, `input_height` | taCV/DMA preprocessing; `224×224` input size |
| Postprocess | `postprocess.top_k`, `labels_path` | Number of output classes and label file |
| Debug | `enable_profile` | Prints timings for input copy, taCV, tensorize, inference, post-processing |

Scene yaml files are full instances of the above — one file per use case:

| `params_file` | `input_msg` | Model |
|---------------|-------------|-------|
| `ai_infer_mobilenet.yaml` | `tb_img` | Mobilenetv2 (online default) |
| `sensor_image_mobilenet.yaml` | `sensor_image` | Mobilenetv2 |
| `file_mobilenet.yaml` | `file` | Mobilenetv2 (offline default) |

Merge order: `params_file` yaml → launch overrides (`input_topic`, `input_msg`, etc.; non-empty only). See `config/README.md`.

## Launch files

| Launch | Purpose |
|--------|---------|
| `file_cli.launch.py` | Offline file mode |
| `mobilenet.launch.py` | Online single node |
| `cam_mobilenet_webviz_demo.launch.py` | Camera + mobilenet + Web joint test |

List arguments: `ros2 launch topsbot_mobilenet_cls <launch> -s`

---

## License

Apache License 2.0 — see [LICENSE](./LICENSE) and [NOTICE](./NOTICE).

Contributions welcome; see [CONTRIBUTING.md](./CONTRIBUTING.md).