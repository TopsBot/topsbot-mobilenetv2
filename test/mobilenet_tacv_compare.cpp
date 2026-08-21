// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "topsbot_mobilenet_cls/dma_buffer.hpp"
#include "topsbot_mobilenet_cls/dma_preprocess.hpp"
#include "topsbot_cv/dmabuf_allocator.hpp"
#include "topsbot_cv/ta_image.hpp"

namespace
{

std::vector<uint8_t> read_binary(const std::string & path)
{
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  return std::vector<uint8_t>(
    std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <input.nv12> <width> <height>\n";
    return 1;
  }
  const int width = std::stoi(argv[2]);
  const int height = std::stoi(argv[3]);
  if (width <= 0 || height <= 0 || (width % 2) != 0 || (height % 2) != 0) {
    std::cerr << "error: NV12 dimensions must be positive and even\n";
    return 1;
  }
  const size_t expected = static_cast<size_t>(width) * static_cast<size_t>(height) * 3U / 2U;
  const auto bytes = read_binary(argv[1]);
  if (bytes.size() != expected) {
    std::cerr << "error: expected " << expected << " NV12 bytes, got " << bytes.size() << '\n';
    return 1;
  }

  topsbot_cv::DmabufAllocator allocator;
  if (!allocator.get()) {
    std::cerr << "error: carveout dmabuf heap is unavailable\n";
    return 1;
  }
  auto input = topsbot_cv::TaImage::create_nv12(allocator, width, height);
  if (input.fd() < 0 ||
    !topsbot_mobilenet_cls::upload_nv12_dma(input, bytes.data(), width, height))
  {
    std::cerr << "error: failed to upload NV12 input\n";
    return 1;
  }

  constexpr int output_width = 224;
  constexpr int output_height = 224;
  ta_cv_rect_t crop{};
  crop.crop_w = width;
  crop.crop_h = height;
  ta_cv_resize_t resize{};
  resize.in_width = width;
  resize.in_height = height;
  resize.out_width = output_width;
  resize.out_height = output_height;
  ta_cv_resize_image_t resize_image{};
  resize_image.resize_img_attr = &resize;
  resize_image.interpolation = TA_CV_INTER_LINEAR;
  topsbot_cv::TaImage tacv_output;
  if (topsbot_mobilenet_cls::image_csc_hardware(
      input, crop, resize_image, CSC_YCbCr2RGB_BT601, FORMAT_RGB_PLANAR,
      allocator, tacv_output) != TACV_SUCCESS || !tacv_output.mapped())
  {
    std::cerr << "error: taCV hardware resize/CSC failed\n";
    return 1;
  }
  if (!topsbot_mobilenet_cls::dmabuf_cpu_sync(
      tacv_output.fd(), topsbot_mobilenet_cls::CpuAccess::Read, true))
  {
    std::cerr << "error: failed to sync taCV output for CPU read\n";
    return 1;
  }

  cv::Mat nv12(height * 3 / 2, width, CV_8UC1, const_cast<uint8_t *>(bytes.data()));
  cv::Mat reference_rgb;
  cv::cvtColor(nv12, reference_rgb, cv::COLOR_YUV2RGB_NV12);
  cv::Mat resized_reference_rgb;
  cv::resize(
    reference_rgb, resized_reference_rgb, cv::Size(output_width, output_height),
    0.0, 0.0, cv::INTER_LINEAR);

  const size_t pixels = static_cast<size_t>(output_width) * output_height;
  const auto * tacv = static_cast<const uint8_t *>(tacv_output.mapped());
  std::vector<unsigned int> errors;
  errors.reserve(pixels * 3U);
  uint64_t absolute_sum = 0U;
  for (int row = 0; row < output_height; ++row) {
    const auto * reference = resized_reference_rgb.ptr<cv::Vec3b>(row);
    for (int column = 0; column < output_width; ++column) {
      const size_t pixel = static_cast<size_t>(row) * output_width + column;
      for (size_t channel = 0; channel < 3U; ++channel) {
        const auto difference = static_cast<unsigned int>(std::abs(
          static_cast<int>(tacv[channel * pixels + pixel]) -
          static_cast<int>(reference[column][channel])));
        errors.push_back(difference);
        absolute_sum += difference;
      }
    }
  }
  const bool sync_ok = topsbot_mobilenet_cls::dmabuf_cpu_sync(
    tacv_output.fd(), topsbot_mobilenet_cls::CpuAccess::Read, false);
  if (!sync_ok) {
    std::cerr << "error: failed to end taCV output CPU sync\n";
    return 1;
  }

  std::sort(errors.begin(), errors.end());
  const size_t p99_index = std::min(
    errors.size() - 1U, (errors.size() * 99U + 99U) / 100U - 1U);
  const double mean_absolute_error =
    static_cast<double>(absolute_sum) / static_cast<double>(errors.size());
  const unsigned int p99 = errors[p99_index];
  std::cout << std::fixed << std::setprecision(4)
            << "pixels=" << pixels << " channels=3"
            << " mae=" << mean_absolute_error
            << " p99=" << p99
            << " max=" << errors.back() << '\n';
  if (mean_absolute_error > 2.0 || p99 > 6U) {
    std::cerr << "FAIL: required MAE <= 2 and P99 <= 6\n";
    return 2;
  }
  std::cout << "PASS: taCV/OpenCV RGB threshold satisfied\n";
  return 0;
}
