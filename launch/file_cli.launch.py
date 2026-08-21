# Copyright 2026 TOPSBOT
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration

from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

_launch_dir = os.path.dirname(os.path.abspath(__file__))
if _launch_dir not in sys.path:
    sys.path.insert(0, _launch_dir)
from model_utils import (  # noqa: E402
    ensure_mobilenet_models,
    load_params,
    param_value,
    resolve_params_file,
    resolve_resource_path,
)


def _launch_setup(context):
    package_share = get_package_share_directory("topsbot_mobilenet_cls")
    params_path = resolve_params_file(
        package_share, LaunchConfiguration("params_file").perform(context))
    params = load_params(params_path)
    requested_model = LaunchConfiguration("model_path").perform(context)
    model_path = ensure_mobilenet_models(context, params, requested_model)

    image_file = LaunchConfiguration("image_file").perform(context)
    output_file = LaunchConfiguration("output_image_file").perform(context) or str(
        param_value(params, "output_image_file", "/tmp/topsbot_ai_infer/out_mobilenetv2.jpg"))
    labels_path = resolve_resource_path(str(param_value(
        params,
        "postprocess.labels_path",
        "package://topsbot_mobilenet_cls/config/imagenet_labels.txt")))

    executable = os.path.join(
        get_package_prefix("topsbot_mobilenet_cls"),
        "lib", "topsbot_mobilenet_cls", "mobilenet_file_cli")
    command = [
        executable,
        image_file,
        output_file,
        "--model-path", model_path,
        "--labels-path", labels_path,
        "--top-k", str(int(param_value(params, "postprocess.top_k", 5))),
        "--device-id", str(int(param_value(params, "device_id", 0))),
        "--csc-type", str(param_value(params, "preprocess.csc_type", "ycbcr2rgb_bt601")),
    ]
    return [
        LogInfo(msg=f"[file_cli] params_file: {params_path}"),
        ExecuteProcess(cmd=command, output="screen"),
    ]


def generate_launch_description():
    default_image = PathJoinSubstitution([
        FindPackageShare("topsbot_mobilenet_cls"),
        "imgs",
        "test.jpg",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value="file_mobilenet.yaml"),
        DeclareLaunchArgument(
            "image_file",
            default_value=default_image,
            description="Input JPEG (default: share/topsbot_mobilenet_cls/imgs/test.jpg)",
        ),
        DeclareLaunchArgument("output_image_file", default_value=""),
        DeclareLaunchArgument("model_path", default_value=""),
        DeclareLaunchArgument("models_dir", default_value="~/topsbot/models"),
        DeclareLaunchArgument("auto_download_models", default_value="true"),
        OpaqueFunction(function=_launch_setup),
    ])
