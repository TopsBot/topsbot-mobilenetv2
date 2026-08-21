// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#ifndef TOPSBOT_MOBILENET_CLS__MOBILENET_UTILS_HPP_
#define TOPSBOT_MOBILENET_CLS__MOBILENET_UTILS_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <ta-runtime-api.h>

#include "topsbot_ai_infer/inference_backend.hpp"

namespace topsbot_mobilenet_cls
{

using topsbot_ai_infer::ClassifierMeta;
using topsbot_ai_infer::ModelConfig;

size_t tensor_element_count(const taconn_inout_attr_t & attr);
size_t tensor_byte_size(const taconn_inout_attr_t & attr);

uint16_t float_to_half(float value);
float half_to_float(uint16_t value);

bool pack_planar_u8_tensor(
  const uint8_t * planar,
  bool source_is_bgr,
  int width,
  int height,
  const ModelConfig & config,
  const taconn_inout_attr_t & input_attr,
  void * output,
  size_t output_size,
  std::string & error);

bool decode_top_k(
  const taconn_buffer_t & output,
  const taconn_inout_attr_t & output_attr,
  int top_k,
  const std::vector<std::string> & labels,
  const std::string & model_name,
  int64_t timestamp_ns,
  std::vector<ClassifierMeta> & results,
  std::string & error);

bool load_labels_file(
  const std::string & path,
  std::vector<std::string> & labels,
  std::string & error);

}  // namespace topsbot_mobilenet_cls

#endif  // TOPSBOT_MOBILENET_CLS__MOBILENET_UTILS_HPP_
