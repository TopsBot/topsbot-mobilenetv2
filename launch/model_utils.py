# Copyright 2026 TOPSBOT
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess

import yaml
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration


def resolve_params_file(package_share: str, params_file: str) -> str:
    path = params_file if os.path.isabs(params_file) else os.path.join(
        package_share, "config", params_file)
    if not os.path.isfile(path):
        raise RuntimeError(f"topsbot_mobilenet_cls params file not found: {path}")
    return path


def load_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("/**", {}).get("ros__parameters", data)


def param_value(params: dict, key: str, default=None):
    if key in params:
        return params[key]
    value = params
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def resolve_resource_path(path: str) -> str:
    if not path.startswith("package://"):
        return os.path.expanduser(path)
    resource = path[len("package://"):]
    package, separator, relative = resource.partition("/")
    if not package or not separator or not relative:
        raise RuntimeError(f"invalid package resource path: {path}")
    return os.path.join(get_package_share_directory(package), relative)


def ensure_mobilenet_models(context, params: dict, explicit_model_path: str = "") -> str:
    backend = str(param_value(params, "model.backend", "")).lower()
    if backend not in ("mobilenet", "mobilenetv2"):
        raise RuntimeError(f"expected a MobileNet backend, got: {backend}")
    package_share = get_package_share_directory("topsbot_mobilenet_cls")
    script = os.path.join(package_share, "scripts", "ensure_models.sh")
    models_dir = os.path.expanduser(LaunchConfiguration("models_dir").perform(context))
    configured_path = explicit_model_path or str(param_value(params, "model.path", ""))
    configured_path = os.path.expanduser(configured_path)
    model_name = os.path.basename(configured_path) or "mobilenetv2_int8.nb"

    # An explicit/custom existing path must be honored. The model manifest is
    # only used for the conventional models directory.
    if explicit_model_path and os.path.isfile(configured_path):
        return configured_path
    default_models_dir = os.path.expanduser("~/topsbot/models")
    if (not explicit_model_path and models_dir == default_models_dir and
            configured_path and os.path.isfile(configured_path)):
        return configured_path
    configured_parent = os.path.dirname(configured_path)
    if configured_path and configured_parent not in (default_models_dir, models_dir):
        raise RuntimeError(f"MobileNet model file not found: {configured_path}")

    auto_download = LaunchConfiguration("auto_download_models").perform(context)
    command = [
        script,
        "--models-dir", models_dir,
        "--auto-download", auto_download,
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"MobileNet model is missing under {models_dir}; "
            "run download_models.sh or enable auto_download_models")
    resolved_path = os.path.join(models_dir, model_name)
    if not os.path.isfile(resolved_path):
        raise RuntimeError(f"MobileNet model file not found after model check: {resolved_path}")
    return resolved_path
