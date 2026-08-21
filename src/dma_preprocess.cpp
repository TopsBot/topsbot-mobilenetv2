// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#include "topsbot_mobilenet_cls/dma_preprocess.hpp"

#include <cstring>

namespace topsbot_mobilenet_cls
{

namespace
{
bool is_sof_marker(const uint8_t marker)
{
  return (marker >= 0xC0 && marker <= 0xC3) ||
         (marker >= 0xC5 && marker <= 0xCB) ||
         marker == 0xC9;
}

bool parse_jpeg_dimensions(
  const uint8_t * jpeg_data,
  const size_t jpeg_size,
  int & width,
  int & height)
{
  if (!jpeg_data || jpeg_size < 4U) {
    return false;
  }

  size_t position = jpeg_data[0] == 0xFF && jpeg_data[1] == 0xD8 ? 2U : 0U;
  while (position + 1U < jpeg_size) {
    if (jpeg_data[position] != 0xFF) {
      ++position;
      continue;
    }
    ++position;
    while (position < jpeg_size && jpeg_data[position] == 0xFF) {
      ++position;
    }
    if (position >= jpeg_size) {
      break;
    }

    const uint8_t marker = jpeg_data[position++];
    if (marker == 0xD9) {
      break;
    }
    if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
      continue;
    }
    if (position + 2U > jpeg_size) {
      break;
    }

    const size_t segment_length =
      (static_cast<size_t>(jpeg_data[position]) << 8U) | jpeg_data[position + 1U];
    if (segment_length < 2U || position + segment_length > jpeg_size) {
      break;
    }
    if (is_sof_marker(marker) && segment_length >= 7U) {
      height = (static_cast<int>(jpeg_data[position + 3U]) << 8) |
        jpeg_data[position + 4U];
      width = (static_cast<int>(jpeg_data[position + 5U]) << 8) |
        jpeg_data[position + 6U];
      return width > 0 && height > 0;
    }
    position += segment_length;
  }
  return false;
}
}  // namespace

bool upload_nv12_dma(
  topsbot_cv::TaImage & image,
  const uint8_t * source,
  const int width,
  const int height)
{
  const size_t byte_size = static_cast<size_t>(width) * static_cast<size_t>(height) * 3U / 2U;
  if (!image.is_nv12() || !source || width <= 0 || height <= 0 ||
    width != image.width() || height != image.height() ||
    image.mapped() == nullptr || image.byte_size() < byte_size ||
    !dmabuf_cpu_sync(image.fd(), CpuAccess::Write, true))
  {
    return false;
  }
  std::memcpy(image.mapped(), source, byte_size);
  return dmabuf_cpu_sync(image.fd(), CpuAccess::Write, false);
}

bool upload_packed_dma(
  topsbot_cv::TaImage & image,
  const uint8_t * source,
  const int width,
  const int height,
  const int step)
{
  if ((image.format() != FORMAT_RGB_PACKED && image.format() != FORMAT_BGR_PACKED) ||
    !source || width <= 0 || height <= 0 || width != image.width() ||
    height != image.height() || step < width * 3 || image.mapped() == nullptr ||
    !dmabuf_cpu_sync(image.fd(), CpuAccess::Write, true))
  {
    return false;
  }

  auto * destination = static_cast<uint8_t *>(image.mapped());
  const size_t row_size = static_cast<size_t>(width) * 3U;
  for (int row = 0; row < height; ++row) {
    std::memcpy(
      destination + static_cast<size_t>(row) * row_size,
      source + static_cast<size_t>(row) * static_cast<size_t>(step),
      row_size);
  }
  return dmabuf_cpu_sync(image.fd(), CpuAccess::Write, false);
}

bool decode_jpeg_bytes_reuse(
  topsbot_cv::DmabufAllocator & allocator,
  const uint8_t * jpeg_data,
  const size_t jpeg_size,
  DmaBuffer & compressed_input,
  topsbot_cv::TaImage & nv12_output,
  const int width_hint,
  const int height_hint)
{
  if (!allocator.get() || !jpeg_data || jpeg_size == 0U) {
    return false;
  }

  int width = width_hint;
  int height = height_hint;
  if ((width <= 0 || height <= 0) &&
    !parse_jpeg_dimensions(jpeg_data, jpeg_size, width, height))
  {
    return false;
  }
  if (width <= 0 || height <= 0 ||
    !compressed_input.allocate_at_least(allocator, jpeg_size) ||
    !compressed_input.begin_cpu_access(CpuAccess::Write))
  {
    return false;
  }
  std::memcpy(compressed_input.mapped(), jpeg_data, jpeg_size);
  if (!compressed_input.end_cpu_access()) {
    return false;
  }

  if (nv12_output.fd() < 0 || nv12_output.width() != width ||
    nv12_output.height() != height || !nv12_output.is_nv12())
  {
    nv12_output = topsbot_cv::TaImage::create_nv12(allocator, width, height);
  }
  if (nv12_output.fd() < 0 ||
    ta_cv_image_jpeg_dec(
      jpeg_size, static_cast<unsigned int>(compressed_input.fd()),
      &nv12_output.native()) != TACV_SUCCESS)
  {
    return false;
  }
  nv12_output.refresh_from_native(FORMAT_NV12);
  return nv12_output.fd() >= 0 && nv12_output.mapped() != nullptr;
}

tacv_status_t image_csc_hardware(
  const topsbot_cv::TaImage & input,
  const ta_cv_rect_t & crop_rect,
  const ta_cv_resize_image_t & resize_attr,
  const csc_type_t csc_type,
  const ta_image_format_ext_t destination_format,
  topsbot_cv::DmabufAllocator & allocator,
  topsbot_cv::TaImage & output)
{
  if (!resize_attr.resize_img_attr) {
    return TACV_ILLEGAL_PARAMETER;
  }

  const int destination_width = resize_attr.resize_img_attr->out_width;
  const int destination_height = resize_attr.resize_img_attr->out_height;
  if (destination_width <= 0 || destination_height <= 0 ||
    crop_rect.crop_w <= 0 || crop_rect.crop_h <= 0)
  {
    return TACV_ILLEGAL_PARAMETER;
  }

  if (output.fd() < 0 || output.width() != destination_width ||
    output.height() != destination_height || output.format() != destination_format)
  {
    output = topsbot_cv::TaImage::create(
      allocator, destination_format, destination_width, destination_height);
  }
  if (output.fd() < 0) {
    return TACV_NOMEM;
  }

  ta_cv_rect_t crop = crop_rect;
  const auto status = ta_cv_image_csc_convert_to_hardware(
    input.native(), output.native(), crop,
    const_cast<ta_cv_resize_image_t *>(&resize_attr), csc_type);
  if (status == TACV_SUCCESS) {
    output.refresh_from_native(destination_format);
  }
  return status;
}

}  // namespace topsbot_mobilenet_cls
