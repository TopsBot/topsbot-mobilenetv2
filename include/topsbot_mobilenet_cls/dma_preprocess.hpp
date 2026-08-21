// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#ifndef TOPSBOT_MOBILENET_CLS__DMA_PREPROCESS_HPP_
#define TOPSBOT_MOBILENET_CLS__DMA_PREPROCESS_HPP_

#include <cstddef>
#include <cstdint>

#include <ta_cv_api_ext_c.h>

#include "topsbot_mobilenet_cls/dma_buffer.hpp"
#include "topsbot_cv/dmabuf_allocator.hpp"
#include "topsbot_cv/ta_image.hpp"

namespace topsbot_mobilenet_cls
{

bool upload_nv12_dma(
  topsbot_cv::TaImage & image,
  const uint8_t * source,
  int width,
  int height);

bool upload_packed_dma(
  topsbot_cv::TaImage & image,
  const uint8_t * source,
  int width,
  int height,
  int step);

bool decode_jpeg_bytes_reuse(
  topsbot_cv::DmabufAllocator & allocator,
  const uint8_t * jpeg_data,
  size_t jpeg_size,
  DmaBuffer & compressed_input,
  topsbot_cv::TaImage & nv12_output,
  int width_hint = 0,
  int height_hint = 0);

tacv_status_t image_csc_hardware(
  const topsbot_cv::TaImage & input,
  const ta_cv_rect_t & crop_rect,
  const ta_cv_resize_image_t & resize_attr,
  csc_type_t csc_type,
  ta_image_format_ext_t destination_format,
  topsbot_cv::DmabufAllocator & allocator,
  topsbot_cv::TaImage & output);

}  // namespace topsbot_mobilenet_cls

#endif  // TOPSBOT_MOBILENET_CLS__DMA_PREPROCESS_HPP_
