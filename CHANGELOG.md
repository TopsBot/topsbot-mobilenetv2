# Changelog

All notable changes to this package are documented here.

## 0.1.1 (2026-09-24)

### Fixed

- Link taCV codec libraries and declare libtacv system dependencies.

### Changed

- Update script location for release tooling.

## 0.1.0 (2026-08-20)

### Added

- MobileNetV2 taCV/DMA backend plugin (`topsbot_mobilenet_cls`) for `topsbot_ai_infer` on MES20 (NPU + dmabuf).
- Online launch (`mobilenet.launch.py`) and camera + WebViz demo (`cam_mobilenet_webviz_demo.launch.py`).
- Offline file CLI (`file_cli.launch.py`) for single-JPG classification with Top-5 overlay.
- Scene configs for file, sensor image, and `topsbot_ai_infer` plugin paths; ImageNet labels and model manifest.
- Model auto-download via `scripts/download_models.sh` / `ensure_models.sh` (`~/topsbot/models/`).
- Quickstart, bilingual README files, config reference, and unit tests.

### Changed

- Keep DMA helpers (`dma_buffer`, `dma_preprocess`) package-local instead of leaking them as a public API.
