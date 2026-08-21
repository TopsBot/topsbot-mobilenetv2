# Copyright 2026 TOPSBOT
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_launch_dir = os.path.dirname(os.path.abspath(__file__))
if _launch_dir not in sys.path:
    sys.path.insert(0, _launch_dir)
from model_utils import (  # noqa: E402
    ensure_mobilenet_models,
    load_params,
    resolve_params_file,
)


def _launch_setup(context):
    package_share = get_package_share_directory("topsbot_mobilenet_cls")
    params_path = resolve_params_file(
        package_share, LaunchConfiguration("params_file").perform(context))
    params = load_params(params_path)
    requested_model = LaunchConfiguration("model_path").perform(context)
    model_path = ensure_mobilenet_models(context, params, requested_model)

    overrides = {"model.path": model_path}
    for launch_name, parameter_name in (
        ("input_topic", "input_topic"),
        ("input_msg", "input_msg"),
        ("tb_img_profile", "tb_img_profile"),
        ("classifications_topic", "classifications_topic"),
    ):
        value = LaunchConfiguration(launch_name).perform(context)
        if value:
            overrides[parameter_name] = value

    return [
        Node(
            package="topsbot_ai_infer",
            executable="ai_infer_node",
            name="mobilenet_classifier",
            output="screen",
            parameters=[params_path, overrides],
        )
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value="ai_infer_mobilenet.yaml"),
        DeclareLaunchArgument("input_topic", default_value=""),
        DeclareLaunchArgument("input_msg", default_value=""),
        DeclareLaunchArgument("tb_img_profile", default_value=""),
        DeclareLaunchArgument("classifications_topic", default_value=""),
        DeclareLaunchArgument("model_path", default_value=""),
        DeclareLaunchArgument("models_dir", default_value="~/topsbot/models"),
        DeclareLaunchArgument("auto_download_models", default_value="true"),
        OpaqueFunction(function=_launch_setup),
    ])
