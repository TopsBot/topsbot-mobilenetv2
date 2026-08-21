// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "topsbot_mobilenet_cls/mobilenet_utils.hpp"
#include "topsbot_ai_infer/result_messages.hpp"

namespace
{

taconn_inout_attr_t make_attr(
  const unsigned int data_format, const unsigned int quant_format,
  const std::vector<unsigned long> & dimensions)
{
  taconn_inout_attr_t attr{};
  attr.data_format = data_format;
  attr.quant_format = quant_format;
  attr.dim_count = static_cast<unsigned int>(dimensions.size());
  for (size_t index = 0; index < dimensions.size(); ++index) {
    attr.dim_size[index] = dimensions[index];
  }
  return attr;
}

TEST(MobileNetUtils, HalfRoundTrip)
{
  for (const float input : {0.0F, -0.0F, 1.0F, -2.0F, 0.33325F, 65504.0F}) {
    const float output = topsbot_mobilenet_cls::half_to_float(
      topsbot_mobilenet_cls::float_to_half(input));
    EXPECT_NEAR(output, input, std::max(0.0005F, std::abs(input) * 0.001F));
  }
}

TEST(MobileNetUtils, PacksBgrPlanarAsRgbNchw)
{
  // B plane, G plane, R plane; two pixels per plane.
  const uint8_t bgr_planar[] = {10U, 11U, 20U, 21U, 30U, 31U};
  topsbot_ai_infer::ModelConfig config;
  config.input_scale = 1.0F;
  config.mean = {0.0F, 0.0F, 0.0F};
  config.std = {1.0F, 1.0F, 1.0F};
  const auto attr = make_attr(
    TACONN_DATA_FORMAT_FP32, TACONN_QNT_TYPE_NONE, {1UL, 3UL, 1UL, 2UL});
  float tensor[6]{};
  std::string error;
  ASSERT_TRUE(topsbot_mobilenet_cls::pack_planar_u8_tensor(
      bgr_planar, true, 2, 1, config, attr, tensor, sizeof(tensor), error)) << error;
  EXPECT_FLOAT_EQ(tensor[0], 30.0F);
  EXPECT_FLOAT_EQ(tensor[1], 31.0F);
  EXPECT_FLOAT_EQ(tensor[2], 20.0F);
  EXPECT_FLOAT_EQ(tensor[3], 21.0F);
  EXPECT_FLOAT_EQ(tensor[4], 10.0F);
  EXPECT_FLOAT_EQ(tensor[5], 11.0F);
}

TEST(MobileNetUtils, AppliesImageNetNormalizationAndFp16Conversion)
{
  const uint8_t rgb_planar[] = {255U, 128U, 0U};
  topsbot_ai_infer::ModelConfig config;
  config.input_scale = 1.0F / 255.0F;
  config.mean = {0.485F, 0.456F, 0.406F};
  config.std = {0.229F, 0.224F, 0.225F};
  const auto attr = make_attr(
    TACONN_DATA_FORMAT_FP16, TACONN_QNT_TYPE_NONE, {1UL, 3UL, 1UL, 1UL});
  uint16_t tensor[3]{};
  std::string error;
  ASSERT_TRUE(topsbot_mobilenet_cls::pack_planar_u8_tensor(
      rgb_planar, false, 1, 1, config, attr, tensor, sizeof(tensor), error)) << error;
  EXPECT_NEAR(
    topsbot_mobilenet_cls::half_to_float(tensor[0]), (1.0F - 0.485F) / 0.229F,
    0.002F);
  EXPECT_NEAR(
    topsbot_mobilenet_cls::half_to_float(tensor[1]),
    (128.0F / 255.0F - 0.456F) / 0.224F, 0.002F);
  EXPECT_NEAR(
    topsbot_mobilenet_cls::half_to_float(tensor[2]), -0.406F / 0.225F, 0.002F);
}

TEST(MobileNetUtils, AffineInt8QuantizationSaturates)
{
  const uint8_t rgb_planar[] = {0U, 255U, 0U, 255U, 0U, 255U};
  topsbot_ai_infer::ModelConfig config;
  config.input_scale = 1.0F;
  config.mean = {0.0F, 0.0F, 0.0F};
  config.std = {1.0F, 1.0F, 1.0F};
  auto attr = make_attr(
    TACONN_DATA_FORMAT_INT8, TACONN_QNT_TYPE_ASYMMETRIC, {1UL, 3UL, 1UL, 2UL});
  attr.quant_data.affine.tf_scale = 1.0F;
  attr.quant_data.affine.tf_zero_point = 0;
  int8_t tensor[6]{};
  std::string error;
  ASSERT_TRUE(topsbot_mobilenet_cls::pack_planar_u8_tensor(
      rgb_planar, false, 2, 1, config, attr, tensor, sizeof(tensor), error)) << error;
  EXPECT_EQ(tensor[0], 0);
  EXPECT_EQ(tensor[1], 127);
  EXPECT_EQ(tensor[3], 127);
  EXPECT_EQ(tensor[5], 127);
}

TEST(MobileNetUtils, AffineUint8QuantizationSaturates)
{
  const uint8_t rgb_planar[] = {0U, 255U, 0U, 255U, 0U, 255U};
  topsbot_ai_infer::ModelConfig config;
  config.input_scale = 1.0F;
  config.mean = {200.0F, 200.0F, 200.0F};
  config.std = {1.0F, 1.0F, 1.0F};
  auto attr = make_attr(
    TACONN_DATA_FORMAT_UINT8, TACONN_QNT_TYPE_ASYMMETRIC, {1UL, 3UL, 1UL, 2UL});
  attr.quant_data.affine.tf_scale = 0.1F;
  attr.quant_data.affine.tf_zero_point = 128;
  uint8_t tensor[6]{};
  std::string error;
  ASSERT_TRUE(topsbot_mobilenet_cls::pack_planar_u8_tensor(
      rgb_planar, false, 2, 1, config, attr, tensor, sizeof(tensor), error)) << error;
  EXPECT_EQ(tensor[0], 0U);
  EXPECT_EQ(tensor[1], 255U);
  EXPECT_EQ(tensor[2], 0U);
  EXPECT_EQ(tensor[3], 255U);
}

TEST(MobileNetUtils, StableSoftmaxReturnsOrderedTopK)
{
  float logits[] = {1000.0F, 1003.0F, 1002.0F, 999.0F};
  taconn_buffer_t output{};
  output.data = logits;
  output.size = sizeof(logits);
  const auto attr = make_attr(
    TACONN_DATA_FORMAT_FP32, TACONN_QNT_TYPE_NONE, {1UL, 4UL});
  const std::vector<std::string> labels{"zero", "one", "two", "three"};
  std::vector<topsbot_ai_infer::ClassifierMeta> results;
  std::string error;
  ASSERT_TRUE(topsbot_mobilenet_cls::decode_top_k(
      output, attr, 3, labels, "test", 10, results, error)) << error;
  ASSERT_EQ(results.size(), 3U);
  EXPECT_EQ(results[0].attr_id, 1);
  EXPECT_EQ(results[1].attr_id, 2);
  EXPECT_EQ(results[2].attr_id, 0);
  EXPECT_GT(results[0].confidence, results[1].confidence);
  EXPECT_GT(results[1].confidence, results[2].confidence);
}

TEST(MobileNetUtils, DequantizesAffineInt8OutputBeforeTopK)
{
  int8_t logits[] = {-10, 10, 0};
  taconn_buffer_t output{};
  output.data = logits;
  output.size = sizeof(logits);
  auto attr = make_attr(
    TACONN_DATA_FORMAT_INT8, TACONN_QNT_TYPE_ASYMMETRIC, {1UL, 3UL});
  attr.quant_data.affine.tf_scale = 0.5F;
  attr.quant_data.affine.tf_zero_point = 0;
  std::vector<topsbot_ai_infer::ClassifierMeta> results;
  std::string error;
  ASSERT_TRUE(topsbot_mobilenet_cls::decode_top_k(
      output, attr, 2, {"negative", "positive", "zero"}, "test", 0,
      results, error)) << error;
  ASSERT_EQ(results.size(), 2U);
  EXPECT_EQ(results[0].attr_id, 1);
  EXPECT_EQ(results[1].attr_id, 2);
}

TEST(MobileNetUtils, LoadsLabelsWithoutDroppingClassOrder)
{
  const std::string path = "/tmp/topsbot_mobilenet_labels_test.txt";
  {
    std::ofstream output(path);
    ASSERT_TRUE(output.good());
    output << "class zero\nclass one\nclass two\n";
  }
  std::vector<std::string> labels;
  std::string error;
  EXPECT_TRUE(topsbot_mobilenet_cls::load_labels_file(path, labels, error)) << error;
  ASSERT_EQ(labels.size(), 3U);
  EXPECT_EQ(labels[0], "class zero");
  EXPECT_EQ(labels[2], "class two");
  EXPECT_EQ(std::remove(path.c_str()), 0);
}

TEST(MobileNetPlugin, FactoryLoadsAliasesAndRejectsUnknownBackend)
{
  std::string error;
  auto backend = topsbot_ai_infer::create_inference_backend("MobileNet", &error);
  ASSERT_NE(backend, nullptr) << error;
  EXPECT_STREQ(backend->name(), "mobilenet");
  topsbot_ai_infer::ModelConfig invalid_config;
  invalid_config.backend = "mobilenet";
  invalid_config.model_path = "/path/that/does/not/exist/mobilenetv2.nb";
  EXPECT_FALSE(backend->initialize(invalid_config, error));
  EXPECT_NE(error.find("model file not found"), std::string::npos);
  backend.reset();

  auto alias = topsbot_ai_infer::create_inference_backend("mobilenetv2", &error);
  ASSERT_NE(alias, nullptr) << error;
  alias.reset();

  auto missing = topsbot_ai_infer::create_inference_backend(
    "backend_that_does_not_exist", &error);
  EXPECT_EQ(missing, nullptr);
  EXPECT_FALSE(error.empty());
}

TEST(MobileNetMessages, ConvertsRankedAttributesAndPerformance)
{
  topsbot_ai_infer::InferenceResult result;
  result.model_name = "mobilenetv2";
  result.device_id = 1;
  topsbot_ai_infer::ClassifierMeta first;
  first.attr_id = 42;
  first.label = "class 42";
  first.confidence = 0.75F;
  topsbot_ai_infer::ClassifierMeta second;
  second.attr_id = 7;
  second.label = "class 7";
  second.confidence = 0.20F;
  result.frame.frame_attributes = {first, second};

  std_msgs::msg::Header header;
  header.frame_id = "camera";
  tb_det_msgs::msg::TbPerf perf;
  perf.type = "mobilenet_infer";
  perf.time_ms_duration = 3.5;
  const auto message = topsbot_ai_infer::classification_message_from_result(
    result, header, {perf});
  EXPECT_EQ(message.header.frame_id, "camera");
  EXPECT_EQ(message.model_name, "mobilenetv2");
  EXPECT_EQ(message.device_id, 1);
  ASSERT_EQ(message.results.size(), 2U);
  EXPECT_EQ(message.results[0].class_id, 42);
  EXPECT_EQ(message.results[0].label, "class 42");
  EXPECT_FLOAT_EQ(message.results[0].score, 0.75F);
  ASSERT_EQ(message.perfs.size(), 1U);
  EXPECT_EQ(message.perfs[0].type, "mobilenet_infer");
}

}  // namespace
