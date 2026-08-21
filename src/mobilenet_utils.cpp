// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#include "topsbot_mobilenet_cls/mobilenet_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>

namespace topsbot_mobilenet_cls
{

size_t tensor_element_count(const taconn_inout_attr_t & attr)
{
  size_t count = 1;
  for (unsigned int index = 0; index < attr.dim_count; ++index) {
    if (attr.dim_size[index] == 0) {
      return 0;
    }
    count *= static_cast<size_t>(attr.dim_size[index]);
  }
  return count;
}

namespace
{
size_t element_size(const unsigned int format)
{
  switch (format) {
    case TACONN_DATA_FORMAT_FP32:
      return sizeof(float);
    case TACONN_DATA_FORMAT_FP16:
      return sizeof(uint16_t);
    case TACONN_DATA_FORMAT_UINT8:
    case TACONN_DATA_FORMAT_INT8:
      return sizeof(uint8_t);
    default:
      return 0;
  }
}

float decode_value(
  const void * data, const size_t index, const taconn_inout_attr_t & attr)
{
  switch (attr.data_format) {
    case TACONN_DATA_FORMAT_FP32:
      return static_cast<const float *>(data)[index];
    case TACONN_DATA_FORMAT_FP16:
      return half_to_float(static_cast<const uint16_t *>(data)[index]);
    case TACONN_DATA_FORMAT_UINT8: {
      const float raw = static_cast<float>(static_cast<const uint8_t *>(data)[index]);
      if (attr.quant_format == TACONN_QNT_TYPE_ASYMMETRIC) {
        return (raw - static_cast<float>(attr.quant_data.affine.tf_zero_point)) *
               attr.quant_data.affine.tf_scale;
      }
      return raw;
    }
    case TACONN_DATA_FORMAT_INT8: {
      const float raw = static_cast<float>(static_cast<const int8_t *>(data)[index]);
      if (attr.quant_format == TACONN_QNT_TYPE_ASYMMETRIC) {
        return (raw - static_cast<float>(attr.quant_data.affine.tf_zero_point)) *
               attr.quant_data.affine.tf_scale;
      }
      return raw;
    }
    default:
      return std::numeric_limits<float>::quiet_NaN();
  }
}
}  // namespace

size_t tensor_byte_size(const taconn_inout_attr_t & attr)
{
  return tensor_element_count(attr) * element_size(attr.data_format);
}

uint16_t float_to_half(const float value)
{
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16U) & 0x8000U;
  int32_t exponent = static_cast<int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;
  uint32_t mantissa = bits & 0x7FFFFFU;

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa = (mantissa | 0x800000U) >> static_cast<uint32_t>(1 - exponent);
    if ((mantissa & 0x1000U) != 0U) {
      mantissa += 0x2000U;
    }
    return static_cast<uint16_t>(sign | (mantissa >> 13U));
  }
  if (exponent >= 31) {
    if (((bits >> 23U) & 0xFFU) == 0xFFU && mantissa != 0U) {
      return static_cast<uint16_t>(sign | 0x7C00U | (mantissa >> 13U) | 1U);
    }
    return static_cast<uint16_t>(sign | 0x7C00U);
  }
  if ((mantissa & 0x1000U) != 0U) {
    mantissa += 0x2000U;
    if ((mantissa & 0x800000U) != 0U) {
      mantissa = 0;
      ++exponent;
      if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
      }
    }
  }
  return static_cast<uint16_t>(
    sign | (static_cast<uint32_t>(exponent) << 10U) | (mantissa >> 13U));
}

float half_to_float(const uint16_t value)
{
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16U;
  uint32_t exponent = (value >> 10U) & 0x1FU;
  uint32_t mantissa = value & 0x03FFU;
  uint32_t bits = 0;

  if (exponent == 0U) {
    if (mantissa == 0U) {
      bits = sign;
    } else {
      int32_t unbiased = -14;
      while ((mantissa & 0x0400U) == 0U) {
        mantissa <<= 1U;
        --unbiased;
      }
      mantissa &= 0x03FFU;
      bits = sign |
        (static_cast<uint32_t>(unbiased + 127) << 23U) |
        (mantissa << 13U);
    }
  } else if (exponent == 0x1FU) {
    bits = sign | 0x7F800000U | (mantissa << 13U);
  } else {
    exponent = exponent - 15U + 127U;
    bits = sign | (exponent << 23U) | (mantissa << 13U);
  }

  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

bool pack_planar_u8_tensor(
  const uint8_t * planar,
  const bool source_is_bgr,
  const int width,
  const int height,
  const ModelConfig & config,
  const taconn_inout_attr_t & input_attr,
  void * output,
  const size_t output_size,
  std::string & error)
{
  if (!planar || !output || width <= 0 || height <= 0) {
    error = "invalid planar image or output buffer";
    return false;
  }
  const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
  if (tensor_element_count(input_attr) != pixels * 3U) {
    error = "MobileNet input tensor is not a three-channel NCHW image";
    return false;
  }
  if (tensor_byte_size(input_attr) == 0 || output_size < tensor_byte_size(input_attr)) {
    error = "unsupported or undersized MobileNet input tensor";
    return false;
  }
  for (const float deviation : config.std) {
    if (!std::isfinite(deviation) || deviation == 0.0F) {
      error = "preprocess.std must contain finite non-zero values";
      return false;
    }
  }

  const bool affine = input_attr.quant_format == TACONN_QNT_TYPE_ASYMMETRIC;
  if (!(input_attr.quant_format == TACONN_QNT_TYPE_NONE &&
    (input_attr.data_format == TACONN_DATA_FORMAT_FP32 ||
    input_attr.data_format == TACONN_DATA_FORMAT_FP16)) &&
    !(affine && (input_attr.data_format == TACONN_DATA_FORMAT_UINT8 ||
    input_attr.data_format == TACONN_DATA_FORMAT_INT8)))
  {
    error = "unsupported MobileNet input data/quantization format";
    return false;
  }
  if (affine && (!std::isfinite(input_attr.quant_data.affine.tf_scale) ||
    input_attr.quant_data.affine.tf_scale <= 0.0F))
  {
    error = "invalid MobileNet input affine scale";
    return false;
  }

  for (size_t channel = 0; channel < 3U; ++channel) {
    const size_t source_channel = source_is_bgr ? 2U - channel : channel;
    const uint8_t * source = planar + source_channel * pixels;
    const size_t base = channel * pixels;
    for (size_t index = 0; index < pixels; ++index) {
      const float normalized =
        (static_cast<float>(source[index]) * config.input_scale - config.mean[channel]) /
        config.std[channel];
      if (input_attr.data_format == TACONN_DATA_FORMAT_FP32) {
        static_cast<float *>(output)[base + index] = normalized;
      } else if (input_attr.data_format == TACONN_DATA_FORMAT_FP16) {
        static_cast<uint16_t *>(output)[base + index] = float_to_half(normalized);
      } else {
        const float quantized = normalized / input_attr.quant_data.affine.tf_scale +
          static_cast<float>(input_attr.quant_data.affine.tf_zero_point);
        const long rounded = std::lround(quantized);
        if (input_attr.data_format == TACONN_DATA_FORMAT_UINT8) {
          static_cast<uint8_t *>(output)[base + index] = static_cast<uint8_t>(
            std::clamp<long>(rounded, 0L, 255L));
        } else {
          static_cast<int8_t *>(output)[base + index] = static_cast<int8_t>(
            std::clamp<long>(rounded, -128L, 127L));
        }
      }
    }
  }
  return true;
}

bool decode_top_k(
  const taconn_buffer_t & output,
  const taconn_inout_attr_t & output_attr,
  const int top_k,
  const std::vector<std::string> & labels,
  const std::string & model_name,
  const int64_t timestamp_ns,
  std::vector<ClassifierMeta> & results,
  std::string & error)
{
  results.clear();
  const size_t count = tensor_element_count(output_attr);
  if (!output.data || count == 0 || output.size < tensor_byte_size(output_attr)) {
    error = "invalid MobileNet output buffer";
    return false;
  }
  if (output_attr.data_format != TACONN_DATA_FORMAT_FP32 &&
    output_attr.data_format != TACONN_DATA_FORMAT_FP16 &&
    output_attr.data_format != TACONN_DATA_FORMAT_UINT8 &&
    output_attr.data_format != TACONN_DATA_FORMAT_INT8)
  {
    error = "unsupported MobileNet output data format";
    return false;
  }
  if (!labels.empty() && labels.size() != count) {
    error = "label count does not match MobileNet output classes";
    return false;
  }

  std::vector<float> logits(count);
  for (size_t index = 0; index < count; ++index) {
    logits[index] = decode_value(output.data, index, output_attr);
    if (!std::isfinite(logits[index])) {
      error = "MobileNet output contains non-finite values";
      return false;
    }
  }
  const float maximum = *std::max_element(logits.begin(), logits.end());
  double denominator = 0.0;
  for (float & value : logits) {
    value = std::exp(value - maximum);
    denominator += static_cast<double>(value);
  }
  if (!std::isfinite(denominator) || denominator <= 0.0) {
    error = "MobileNet softmax denominator is invalid";
    return false;
  }

  std::vector<size_t> indexes(count);
  std::iota(indexes.begin(), indexes.end(), 0U);
  const size_t selected = std::min(count, static_cast<size_t>(std::max(1, top_k)));
  std::partial_sort(
    indexes.begin(), indexes.begin() + static_cast<std::ptrdiff_t>(selected), indexes.end(),
    [&logits](const size_t left, const size_t right) {return logits[left] > logits[right];});

  results.reserve(selected);
  for (size_t rank = 0; rank < selected; ++rank) {
    const size_t class_id = indexes[rank];
    ClassifierMeta classification;
    classification.attr_id = static_cast<int>(class_id);
    classification.label = labels.empty() ?
      "class_" + std::to_string(class_id) : labels[class_id];
    classification.confidence = static_cast<float>(
      static_cast<double>(logits[class_id]) / denominator);
    classification.model_name = model_name;
    classification.timestamp_ns = timestamp_ns;
    results.push_back(std::move(classification));
  }
  return true;
}

bool load_labels_file(
  const std::string & path, std::vector<std::string> & labels, std::string & error)
{
  labels.clear();
  std::ifstream input(path);
  if (!input) {
    error = "failed to open labels file: " + path;
    return false;
  }
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (labels.empty() && line.size() >= 3U &&
      static_cast<unsigned char>(line[0]) == 0xEFU &&
      static_cast<unsigned char>(line[1]) == 0xBBU &&
      static_cast<unsigned char>(line[2]) == 0xBFU)
    {
      line.erase(0, 3);
    }
    if (!line.empty()) {
      labels.push_back(line);
    }
  }
  if (labels.empty()) {
    error = "labels file is empty: " + path;
    return false;
  }
  return true;
}

}  // namespace topsbot_mobilenet_cls
