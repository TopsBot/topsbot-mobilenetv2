// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "topsbot_ai_infer/inference_backend.hpp"

namespace
{

void print_usage(const char * executable)
{
  std::cerr
    << "Usage: " << executable << " <input.jpg> <output.jpg> [options]\n"
    << "  --model-path PATH    required .nb model file\n"
    << "  --labels-path PATH   ImageNet labels (default: package labels)\n"
    << "  --top-k INT          number of results (default: 5)\n"
    << "  --device-id INT      NPU core (default: 0)\n"
    << "  --csc-type TYPE      ycbcr2rgb_bt601|bt709 form (default: ycbcr2rgb_bt601)\n";
}

std::string next_argument(int argc, char ** argv, int & index)
{
  if (index + 1 >= argc) {
    return {};
  }
  return argv[++index];
}

std::string expand_user_path(const std::string & path)
{
  if (path.empty() || path.front() != '~') {
    return path;
  }
  const char * user_home = std::getenv("HOME");
  if (!user_home || user_home[0] == '\0') {
    return path;
  }
  if (path.size() == 1U || path[1] == '/') {
    return std::string(user_home) + path.substr(1);
  }
  return path;
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

}  // namespace

int main(int argc, char ** argv)
{
  if (argc < 3) {
    print_usage(argv[0]);
    return 1;
  }

  const std::string input_path = expand_user_path(argv[1]);
  const std::string output_path = expand_user_path(argv[2]);
  topsbot_ai_infer::ModelConfig config;
  config.backend = "mobilenet";
  config.model_name = "mobilenetv2";
  config.model_type = "mobilenetv2";
  config.input_width = 224;
  config.input_height = 224;
  config.preprocess_engine = "tacv_dma";
  config.letterbox = false;
  config.labels_path =
    ament_index_cpp::get_package_share_directory("topsbot_mobilenet_cls") +
    "/config/imagenet_labels.txt";

  for (int index = 3; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--model-path") {
      config.model_path = expand_user_path(next_argument(argc, argv, index));
    } else if (argument == "--labels-path") {
      config.labels_path = expand_user_path(next_argument(argc, argv, index));
    } else if (argument == "--top-k") {
      config.top_k = std::atoi(next_argument(argc, argv, index).c_str());
    } else if (argument == "--device-id") {
      config.device_id = std::atoi(next_argument(argc, argv, index).c_str());
    } else if (argument == "--csc-type") {
      config.csc_type = next_argument(argc, argv, index);
    } else {
      std::cerr << "unknown option: " << argument << '\n';
      print_usage(argv[0]);
      return 1;
    }
  }

  if (config.model_path.empty()) {
    std::cerr << "error: --model-path is required\n";
    return 1;
  }
  if (!std::filesystem::is_regular_file(input_path)) {
    std::cerr << "error: input file not found: " << input_path << '\n';
    return 1;
  }

  auto jpeg = read_binary(input_path);
  if (!jpeg || jpeg->empty()) {
    std::cerr << "error: failed to read input JPEG\n";
    return 1;
  }
  std::string error;
  auto backend = topsbot_ai_infer::create_inference_backend(config.backend, &error);
  if (!backend || !backend->initialize(config, error)) {
    std::cerr << "error: MobileNet initialization failed: " << error << '\n';
    return 1;
  }

  topsbot_ai_infer::InferenceFrame frame;
  frame.owner = jpeg;
  frame.data = jpeg->data();
  frame.data_size = jpeg->size();
  frame.format = topsbot_ai_infer::PixelFormat::Jpeg;
  frame.sequence = 1U;
  frame.source_id = "file";
  const auto result = backend->infer(frame);
  if (!result.success) {
    std::cerr << "error: MobileNet inference failed: " << result.error << '\n';
    return 1;
  }

  cv::Mat visualization = cv::imread(input_path, cv::IMREAD_COLOR);
  if (visualization.empty()) {
    std::cerr << "error: failed to decode visualization image\n";
    return 1;
  }
  int baseline = 0;
  const double font_scale = std::max(0.55, visualization.cols / 1200.0);
  const int line_height = cv::getTextSize(
    "Ag", cv::FONT_HERSHEY_SIMPLEX, font_scale, 1, &baseline).height + 10;
  int y = line_height;
  for (size_t rank = 0; rank < result.frame.frame_attributes.size(); ++rank) {
    const auto & classification = result.frame.frame_attributes[rank];
    std::ostringstream text;
    text << (rank + 1U) << ". " << classification.label << " ["
         << classification.attr_id << "] " << std::fixed << std::setprecision(4)
         << classification.confidence;
    cv::putText(
      visualization, text.str(), cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX,
      font_scale, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(
      visualization, text.str(), cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX,
      font_scale, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    y += line_height;
    std::cout << text.str() << '\n';
  }

  const auto parent = std::filesystem::path(output_path).parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  if (!cv::imwrite(output_path, visualization)) {
    std::cerr << "error: failed to write output image: " << output_path << '\n';
    return 1;
  }
  std::cout << "Timing ms: input_copy=" << result.timing.input_copy_ms
            << " tacv=" << result.timing.accelerator_preprocess_ms
            << " tensorize=" << result.timing.tensorize_ms
            << " infer=" << result.timing.infer_ms
            << " post=" << result.timing.postprocess_ms << '\n'
            << "Wrote " << output_path << '\n';
  return 0;
}
