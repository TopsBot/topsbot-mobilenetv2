# topsbot_mobilenet_cls 快速入门

完整说明见 `README_cn.md`（English: `README.md`）。以下命令均在 MES20 板端执行。

## 1. 离线 file 模式

```bash
# 默认使用 MobileNetV2 和包内测试图片
ros2 launch topsbot_mobilenet_cls file_cli.launch.py

# 修改输入和输出
ros2 launch topsbot_mobilenet_cls file_cli.launch.py \
  image_file:=/path/to/in.jpg \
  output_image_file:=/path/to/out.jpg

# 已预置模型或离线运行时关闭自动下载
ros2 launch topsbot_mobilenet_cls file_cli.launch.py auto_download_models:=false
```

## 2. 单节点模式（在线推理）

```bash
# 默认订阅 /tbmem_img，发布 /classifications
ros2 launch topsbot_mobilenet_cls mobilenet.launch.py

# 多相机 / 换话题 / 跳帧
ros2 launch topsbot_mobilenet_cls mobilenet.launch.py input_topic:=/cam0/tbmem_img
ros2 launch topsbot_mobilenet_cls mobilenet.launch.py frame_skip:=2

# sensor_msgs 路径（须与 usb_cam 成对切换）
ros2 launch topsbot_usb_cam usb_cam.launch.py params_file:=mjpeg2nv12_640x480_sensor.yaml setup_shm:=false
ros2 launch topsbot_mobilenet_cls mobilenet.launch.py params_file:=sensor_image_mobilenet.yaml
```

## 3. 联合测试（相机 + MobileNet + Web）

```bash
source /opt/tbros/humble/setup.bash
#source <install>/setup.bash
ros2 launch topsbot_mobilenet_cls cam_mobilenet_webviz_demo.launch.py
```

浏览器访问 `http://<板卡IP>:8000/` 查看 NV12 画面与 Top-5 分类结果叠加。无 Web：
`enable_webviz:=false`。

**确认链路正常**

```bash
ros2 topic hz /tbmem_img
ros2 topic hz /classifications
ros2 topic echo /classifications --once
```

启动成功时，MobileNet 节点日志类似：

```text
ready: backend=mobilenet model=mobilenetv2 input=tb_img:/tbmem_img workers=1 strategy=least_loaded queue=2 drop=drop_oldest classifications=/classifications
AI worker core=.. healthy=.. in_flight=.. assigned=.. completed=.. failed=.. dropped=.. latency_ewma=..ms
```

## 4. 查看 launch 参数

```bash
ros2 launch topsbot_mobilenet_cls file_cli.launch.py -s
ros2 launch topsbot_mobilenet_cls mobilenet.launch.py -s
ros2 launch topsbot_mobilenet_cls cam_mobilenet_webviz_demo.launch.py -s
```

## 5. 配置与模型

- 场景 yaml：`config/README.md`
- 参数字典：`config/params_reference.yaml`
- 默认模型：`~/topsbot/models/mobilenetv2_int8.nb`
- 备用模型：`~/topsbot/models/mobilenetv2_float16.nb`
- 模型清单：`config/models_manifest.yaml`
