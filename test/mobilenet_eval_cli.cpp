// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "topsbot_ai_infer/inference_backend.hpp"

namespace
{

std::string expand_user_path(const std::string & path)
{
  if (path.empty() || path.front() != '~') {
    return path;
  }
  const char * user_home = std::getenv("HOME");
  if (!user_home || user_home[0] == '\0') {
    return path;
  }
  return (path.size() == 1U || path[1] == '/') ?
    std::string(user_home) + path.substr(1) : path;
}

std::shared_ptr<std::vector<uint8_t>> read_binary(const std::string & path)
{
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return nullptr;
  }
  return std::make_shared<std::vector<uint8_t>>(
    std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string next_argument(int argc, char ** argv, int & index)
{
  return index + 1 < argc ? argv[++index] : std::string{};
}

void usage(const char * executable)
{
  std::cerr
    << "Usage: " << executable << " <validation.txt> --model-path MODEL [options]\n"
    << "Each non-comment line: relative/or/absolute.jpg class_id\n"
    << "  --images-dir DIR    prefix for relative image paths\n"
    << "  --labels-path PATH  ImageNet labels (default: package labels)\n"
    << "  --device-id INT     NPU core (default: 0)\n"
    << "  --limit INT         evaluate at most this many images\n";
}

struct Entry
{
  std::string path;
  int class_id{-1};
};

bool load_entries(const std::string & path, std::vector<Entry> & entries)
{
  std::ifstream input(path);
  if (!input) {
    return false;
  }
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    std::istringstream parser(line);
    Entry entry;
    if (!(parser >> entry.path >> entry.class_id) || entry.class_id < 0 || entry.class_id >= 1000) {
      return false;
    }
    entries.push_back(std::move(entry));
  }
  return !entries.empty();
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc < 2) {
    usage(argv[0]);
    return 1;
  }
  const std::string validation_path = expand_user_path(argv[1]);
  std::string images_dir;
  int limit = 0;
  topsbot_ai_infer::ModelConfig config;
  config.backend = "mobilenet";
  config.model_name = "mobilenetv2";
  config.model_type = "mobilenetv2";
  config.preprocess_engine = "tacv_dma";
  config.input_width = 224;
  config.input_height = 224;
  config.top_k = 5;
  config.labels_path =
    ament_index_cpp::get_package_share_directory("topsbot_mobilenet_cls") +
    "/config/imagenet_labels.txt";

  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--model-path") {
      config.model_path = expand_user_path(next_argument(argc, argv, index));
    } else if (argument == "--labels-path") {
      config.labels_path = expand_user_path(next_argument(argc, argv, index));
    } else if (argument == "--images-dir") {
      images_dir = expand_user_path(next_argument(argc, argv, index));
    } else if (argument == "--device-id") {
      config.device_id = std::atoi(next_argument(argc, argv, index).c_str());
    } else if (argument == "--limit") {
      limit = std::max(0, std::atoi(next_argument(argc, argv, index).c_str()));
    } else {
      usage(argv[0]);
      return 1;
    }
  }
  if (config.model_path.empty()) {
    usage(argv[0]);
    return 1;
  }

  std::vector<Entry> entries;
  if (!load_entries(validation_path, entries)) {
    std::cerr << "error: invalid or empty validation list: " << validation_path << '\n';
    return 1;
  }
  if (limit > 0 && entries.size() > static_cast<size_t>(limit)) {
    entries.resize(static_cast<size_t>(limit));
  }

  std::string error;
  auto backend = topsbot_ai_infer::create_inference_backend(config.backend, &error);
  if (!backend || !backend->initialize(config, error)) {
    std::cerr << "error: MobileNet initialization failed: " << error << '\n';
    return 1;
  }

  size_t top1_correct = 0U;
  size_t top5_correct = 0U;
  double copy_ms = 0.0;
  double tacv_ms = 0.0;
  double tensorize_ms = 0.0;
  double infer_ms = 0.0;
  double post_ms = 0.0;
  for (size_t index = 0; index < entries.size(); ++index) {
    std::string image_path = entries[index].path;
    if (!images_dir.empty() && !image_path.empty() && image_path.front() != '/') {
      image_path = images_dir + "/" + image_path;
    }
    auto jpeg = read_binary(image_path);
    if (!jpeg || jpeg->empty()) {
      std::cerr << "error: cannot read " << image_path << '\n';
      return 2;
    }
    topsbot_ai_infer::InferenceFrame frame;
    frame.owner = jpeg;
    frame.data = jpeg->data();
    frame.data_size = jpeg->size();
    frame.format = topsbot_ai_infer::PixelFormat::Jpeg;
    frame.sequence = index + 1U;
    frame.source_id = "imagenet_validation";
    const auto result = backend->infer(frame);
    if (!result.success || result.frame.frame_attributes.empty()) {
      std::cerr << "error: inference failed for " << image_path << ": " << result.error << '\n';
      return 2;
    }

    if (result.frame.frame_attributes.front().attr_id == entries[index].class_id) {
      ++top1_correct;
    }
    const bool in_top5 = std::any_of(
      result.frame.frame_attributes.begin(), result.frame.frame_attributes.end(),
      [&entry = entries[index]](const topsbot_ai_infer::ClassifierMeta & item) {
        return item.attr_id == entry.class_id;
      });
    top5_correct += in_top5 ? 1U : 0U;
    copy_ms += result.timing.input_copy_ms;
    tacv_ms += result.timing.accelerator_preprocess_ms;
    tensorize_ms += result.timing.tensorize_ms;
    infer_ms += result.timing.infer_ms;
    post_ms += result.timing.postprocess_ms;
    if ((index + 1U) % 100U == 0U || index + 1U == entries.size()) {
      std::cout << "progress " << (index + 1U) << '/' << entries.size() << '\n';
    }
  }

  const double count = static_cast<double>(entries.size());
  std::cout << std::fixed << std::setprecision(4)
            << "images=" << entries.size() << '\n'
            << "top1=" << static_cast<double>(top1_correct) / count << '\n'
            << "top5=" << static_cast<double>(top5_correct) / count << '\n'
            << "avg_ms input_copy=" << copy_ms / count
            << " tacv=" << tacv_ms / count
            << " tensorize=" << tensorize_ms / count
            << " infer=" << infer_ms / count
            << " post=" << post_ms / count << '\n';
  return 0;
}
