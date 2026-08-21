// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#include "topsbot_ai_infer/inference_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <ta-runtime-api.h>
#include <ta_cv_api_ext_c.h>

#include <pluginlib/class_list_macros.hpp>

#include "topsbot_mobilenet_cls/dma_buffer.hpp"
#include "topsbot_mobilenet_cls/dma_preprocess.hpp"
#include "topsbot_mobilenet_cls/mobilenet_utils.hpp"
#include "topsbot_cv/dmabuf_allocator.hpp"
#include "topsbot_cv/format_convert.hpp"
#include "topsbot_cv/ta_image.hpp"

namespace topsbot_mobilenet_cls
{

using topsbot_ai_infer::IInferenceBackend;
using topsbot_ai_infer::InferenceFrame;
using topsbot_ai_infer::InferenceResult;
using topsbot_ai_infer::ModelConfig;
using topsbot_ai_infer::PixelFormat;

namespace
{

using SteadyClock = std::chrono::steady_clock;

float elapsed_ms(const SteadyClock::time_point & begin, const SteadyClock::time_point & end)
{
  return std::chrono::duration<float, std::milli>(end - begin).count();
}

std::mutex runtime_mutex;
size_t runtime_users = 0U;

bool acquire_runtime()
{
  std::lock_guard<std::mutex> lock(runtime_mutex);
  if (runtime_users == 0U && ta_runtime_init() != NNRT_SUCCESS) {
    return false;
  }
  ++runtime_users;
  return true;
}

void release_runtime()
{
  std::lock_guard<std::mutex> lock(runtime_mutex);
  if (runtime_users == 0U) {
    return;
  }
  --runtime_users;
  if (runtime_users == 0U) {
    (void)ta_runtime_deinit();
  }
}

bool is_reference_input_shape(
  const taconn_inout_attr_t & attr, const int width, const int height)
{
  if (attr.dim_count != 4U || tensor_element_count(attr) !=
    static_cast<size_t>(width) * static_cast<size_t>(height) * 3U)
  {
    return false;
  }
  std::vector<unsigned long> dimensions(
    attr.dim_size, attr.dim_size + attr.dim_count);
  std::sort(dimensions.begin(), dimensions.end());
  std::vector<unsigned long> expected{
    1UL, 3UL, static_cast<unsigned long>(width), static_cast<unsigned long>(height)};
  std::sort(expected.begin(), expected.end());
  return dimensions == expected;
}

bool is_supported_tensor(const taconn_inout_attr_t & attr)
{
  if (attr.quant_format == TACONN_QNT_TYPE_NONE) {
    return attr.data_format == TACONN_DATA_FORMAT_FP32 ||
           attr.data_format == TACONN_DATA_FORMAT_FP16;
  }
  return attr.quant_format == TACONN_QNT_TYPE_ASYMMETRIC &&
         (attr.data_format == TACONN_DATA_FORMAT_INT8 ||
         attr.data_format == TACONN_DATA_FORMAT_UINT8) &&
         std::isfinite(attr.quant_data.affine.tf_scale) &&
         attr.quant_data.affine.tf_scale > 0.0F;
}

unsigned int interpolation_code(const std::string & interpolation)
{
  if (interpolation == "nearest") {
    return static_cast<unsigned int>(TA_CV_INTER_NEAREST);
  }
  if (interpolation == "bilinear") {
    return static_cast<unsigned int>(TA_CV_INTER_BILINEAR);
  }
  return static_cast<unsigned int>(TA_CV_INTER_LINEAR);
}

}  // namespace

class MobileNetBackend final : public IInferenceBackend
{
public:
  ~MobileNetBackend() override
  {
    shutdown();
  }

  bool initialize(const ModelConfig & config, std::string & error) override
  {
    shutdown();
    error.clear();
    config_ = config;
    if (config_.preprocess_engine != "tacv_dma") {
      error = "MobileNet preprocess.engine must be tacv_dma";
      return false;
    }
    if (config_.model_path.empty() || !std::filesystem::is_regular_file(config_.model_path)) {
      error = "MobileNet model file not found: " + config_.model_path;
      return false;
    }
    if (config_.input_width != 224 || config_.input_height != 224) {
      error = "reference MobileNetV2 requires a 224x224 input";
      return false;
    }
    const auto csc = topsbot_cv::parse_csc_type(config_.csc_type);
    if (!csc || (*csc != CSC_YCbCr2RGB_BT601 && *csc != CSC_YCbCr2RGB_BT709)) {
      error = "preprocess.csc_type must be ycbcr2rgb_bt601 or ycbcr2rgb_bt709";
      return false;
    }
    if (config_.interpolation != "linear" && config_.interpolation != "bilinear" &&
      config_.interpolation != "nearest")
    {
      error = "preprocess.interpolation must be linear, bilinear, or nearest";
      return false;
    }
    if (config_.top_k <= 0 || config_.top_k > 1000) {
      error = "postprocess.top_k must be between 1 and 1000";
      return false;
    }
    if (!std::isfinite(config_.input_scale) || config_.input_scale <= 0.0F) {
      error = "preprocess.scale must be finite and positive";
      return false;
    }
    for (size_t channel = 0; channel < 3U; ++channel) {
      if (!std::isfinite(config_.mean[channel]) ||
        !std::isfinite(config_.std[channel]) || config_.std[channel] == 0.0F)
      {
        error = "preprocess.mean/std must contain finite values and non-zero std";
        return false;
      }
    }
    if (config_.labels_path.empty()) {
      error = "postprocess.labels_path is required for MobileNet";
      return false;
    }
    if (!allocator_.get()) {
      error = "failed to create carveout dmabuf allocator";
      return false;
    }
    if (!acquire_runtime()) {
      error = "ta_runtime_init failed";
      return false;
    }
    runtime_acquired_ = true;

    if (ta_runtime_load_model_from_file(
        &context_, config_.model_path.c_str(), static_cast<uint32_t>(config_.device_id)) !=
      NNRT_SUCCESS)
    {
      error = "failed to load MobileNet model on NPU core " +
        std::to_string(config_.device_id);
      shutdown();
      return false;
    }

    taconn_input_output_num_t counts{};
    if (ta_runtime_query(&context_, TACONN_QUERY_IN_OUT_NUM, &counts) != NNRT_SUCCESS ||
      counts.input_num != 1U || counts.output_num != 1U)
    {
      error = "MobileNet backend requires exactly one input and one output";
      shutdown();
      return false;
    }
    input_attr_ = {};
    input_attr_.index = 0U;
    output_attr_ = {};
    output_attr_.index = 0U;
    if (ta_runtime_query(&context_, TACONN_QUERY_INPUT_ATTR, &input_attr_) != NNRT_SUCCESS ||
      ta_runtime_query(&context_, TACONN_QUERY_OUTPUT_ATTR, &output_attr_) != NNRT_SUCCESS)
    {
      error = "failed to query MobileNet tensor attributes";
      shutdown();
      return false;
    }
    if (!is_reference_input_shape(input_attr_, config_.input_width, config_.input_height)) {
      error = "MobileNet input tensor must describe 1x3x224x224";
      shutdown();
      return false;
    }
    if (!is_supported_tensor(input_attr_)) {
      error = "MobileNet input must be FP32/FP16 or asymmetric INT8/UINT8";
      shutdown();
      return false;
    }
    if (tensor_element_count(output_attr_) != 1000U) {
      error = "MobileNet output tensor must contain exactly 1000 classes";
      shutdown();
      return false;
    }
    if (!is_supported_tensor(output_attr_)) {
      error = "MobileNet output must be FP32/FP16 or asymmetric INT8/UINT8";
      shutdown();
      return false;
    }

    const size_t input_size = tensor_byte_size(input_attr_);
    const size_t output_size = tensor_byte_size(output_attr_);
    if (input_size == 0U || output_size == 0U ||
      input_size > std::numeric_limits<uint32_t>::max() ||
      output_size > std::numeric_limits<uint32_t>::max())
    {
      error = "unsupported MobileNet tensor data type or size";
      shutdown();
      return false;
    }
    if (!input_buffer_.allocate(allocator_, input_size)) {
      error = "failed to allocate MobileNet input dmabuf";
      shutdown();
      return false;
    }

    uint64_t physical_address = 0;
    size_t physical_length = 0;
    if (!get_physical_address(
        allocator_,
        input_buffer_.fd(), physical_address, physical_length) ||
      physical_length < input_size)
    {
      error = "failed to resolve MobileNet input physical address";
      shutdown();
      return false;
    }
    input_phy_ = {};
    input_phy_.physical_table[0] = physical_address;
    input_phy_.size_table[0] = input_size;
    if (ta_runtime_set_input_pha(&context_, 1U, &input_phy_) != NNRT_SUCCESS) {
      error = "ta_runtime_set_input_pha failed for MobileNet input";
      shutdown();
      return false;
    }

    output_buffer_ = {};
    if (ta_runtime_create_buffer(
        &context_, static_cast<uint32_t>(output_size), &output_buffer_) != NNRT_SUCCESS)
    {
      error = "failed to allocate MobileNet output buffer";
      shutdown();
      return false;
    }
    output_created_ = true;
    if (ta_runtime_set_output(&context_, 1U, &output_buffer_) != NNRT_SUCCESS) {
      error = "failed to bind MobileNet output buffer";
      shutdown();
      return false;
    }

    if (!load_labels_file(config_.labels_path, labels_, error)) {
      shutdown();
      return false;
    }
    if (labels_.size() != 1000U) {
      error = "MobileNet labels file must contain exactly 1000 non-empty lines";
      shutdown();
      return false;
    }

    initialized_ = true;
    return true;
  }

  InferenceResult infer(const InferenceFrame & frame) override
  {
    InferenceResult result;
    result.backend = name();
    result.model_name = config_.model_name;
    result.device_id = config_.device_id;
    result.frame.source_id = frame.source_id;
    result.frame.frame_number = frame.sequence;
    result.frame.timestamp_ns = frame.timestamp_ns;
    result.frame.tags["backend"] = name();
    result.frame.tags["model"] = config_.model_name;
    result.frame.tags["device_id"] = std::to_string(config_.device_id);

    if (!initialized_) {
      result.error = "MobileNet backend is not initialized";
      return result;
    }

    const auto copy_begin = SteadyClock::now();
    const topsbot_cv::TaImage * source_nv12 = nullptr;
    bool source_is_bgr = false;

    if (frame.format == PixelFormat::Nv12) {
      const size_t expected = static_cast<size_t>(frame.width) *
        static_cast<size_t>(frame.height) * 3U / 2U;
      if (!frame.data || frame.width <= 0 || frame.height <= 0 ||
        (frame.width % 2) != 0 || (frame.height % 2) != 0 || frame.data_size < expected)
      {
        result.error = "invalid NV12 input frame";
        return result;
      }
      if (source_nv12_.fd() < 0 || source_nv12_.width() != frame.width ||
        source_nv12_.height() != frame.height)
      {
        source_nv12_ = topsbot_cv::TaImage::create_nv12(
          allocator_, frame.width, frame.height);
      }
      if (source_nv12_.fd() < 0 ||
        !upload_nv12_dma(source_nv12_, frame.data, frame.width, frame.height))
      {
        result.error = "failed to upload NV12 input into dmabuf";
        return result;
      }
      source_nv12 = &source_nv12_;
    } else if (frame.format == PixelFormat::Jpeg) {
      if (!frame.data || frame.data_size == 0U) {
        result.error = "invalid JPEG input";
        return result;
      }
      if (!decode_jpeg_bytes_reuse(
          allocator_, frame.data, frame.data_size, jpeg_input_buffer_, jpeg_nv12_))
      {
        result.error = "hardware JPEG decode failed";
        return result;
      }
      source_nv12 = &jpeg_nv12_;
    } else {
      const int step = frame.step > 0 ? frame.step : frame.width * 3;
      const size_t expected = static_cast<size_t>(step) * static_cast<size_t>(frame.height);
      if (!frame.data || frame.width <= 0 || frame.height <= 0 ||
        step < frame.width * 3 || frame.data_size < expected)
      {
        result.error = "invalid BGR8 input frame";
        return result;
      }
      if (source_bgr_.fd() < 0 || source_bgr_.width() != frame.width ||
        source_bgr_.height() != frame.height)
      {
        source_bgr_ = topsbot_cv::TaImage::create(
          allocator_, FORMAT_BGR_PACKED, frame.width, frame.height);
      }
      if (source_bgr_.fd() < 0 ||
        !upload_packed_dma(source_bgr_, frame.data, frame.width, frame.height, step))
      {
        result.error = "failed to upload BGR8 input into dmabuf";
        return result;
      }
      source_is_bgr = true;
    }
    const auto copy_end = SteadyClock::now();
    result.timing.input_copy_ms = elapsed_ms(copy_begin, copy_end);

    const auto accelerator_begin = SteadyClock::now();
    if (source_nv12 != nullptr) {
      if (!preprocess_nv12(*source_nv12, result.error)) {
        return result;
      }
    } else if (!preprocess_bgr(result.error)) {
      return result;
    }
    const auto accelerator_end = SteadyClock::now();
    result.timing.accelerator_preprocess_ms = elapsed_ms(accelerator_begin, accelerator_end);

    const auto tensorize_begin = SteadyClock::now();
    if (!dmabuf_cpu_sync(planar_.fd(), CpuAccess::Read, true)) {
      result.error = "failed to begin CPU read sync for taCV output";
      return result;
    }
    if (!input_buffer_.begin_cpu_access(CpuAccess::Write)) {
      (void)dmabuf_cpu_sync(planar_.fd(), CpuAccess::Read, false);
      result.error = "failed to begin CPU write sync for model input";
      return result;
    }
    std::string tensor_error;
    const bool packed = pack_planar_u8_tensor(
      static_cast<const uint8_t *>(planar_.mapped()), source_is_bgr,
      config_.input_width, config_.input_height, config_, input_attr_,
      input_buffer_.mapped(), input_buffer_.size(), tensor_error);
    const bool input_sync_ok = input_buffer_.end_cpu_access();
    const bool planar_sync_ok = dmabuf_cpu_sync(planar_.fd(), CpuAccess::Read, false);
    if (!packed || !input_sync_ok || !planar_sync_ok) {
      result.error = !packed ? tensor_error : "failed to end dmabuf CPU access";
      return result;
    }
    const auto tensorize_end = SteadyClock::now();
    result.timing.tensorize_ms = elapsed_ms(tensorize_begin, tensorize_end);
    result.timing.preprocess_ms = result.timing.input_copy_ms +
      result.timing.accelerator_preprocess_ms + result.timing.tensorize_ms;

    const auto infer_begin = SteadyClock::now();
    if (ta_runtime_run_network(&context_) != NNRT_SUCCESS ||
      ta_runtime_invalidate_buffer(&context_, &output_buffer_) != NNRT_SUCCESS)
    {
      result.error = "MobileNet NPU inference failed";
      return result;
    }
    const auto infer_end = SteadyClock::now();
    result.timing.infer_ms = elapsed_ms(infer_begin, infer_end);

    const auto post_begin = SteadyClock::now();
    if (!decode_top_k(
        output_buffer_, output_attr_, config_.top_k, labels_, config_.model_name,
        frame.timestamp_ns, result.frame.frame_attributes, result.error))
    {
      return result;
    }
    const auto post_end = SteadyClock::now();
    result.timing.postprocess_ms = elapsed_ms(post_begin, post_end);
    result.success = true;
    return result;
  }

  const char * name() const noexcept override {return "mobilenet";}

private:
  bool preprocess_nv12(const topsbot_cv::TaImage & source, std::string & error)
  {
    ta_cv_rect_t crop{};
    crop.start_x = 0;
    crop.start_y = 0;
    crop.crop_w = source.width();
    crop.crop_h = source.height();
    ta_cv_resize_t resize{};
    resize.in_width = source.width();
    resize.in_height = source.height();
    resize.out_width = config_.input_width;
    resize.out_height = config_.input_height;
    ta_cv_resize_image_t resize_image{};
    resize_image.resize_img_attr = &resize;
    resize_image.interpolation = interpolation_code(config_.interpolation);

    const auto csc = topsbot_cv::parse_csc_type(config_.csc_type);
    if (!csc || image_csc_hardware(
        source, crop, resize_image, *csc, FORMAT_RGB_PLANAR,
        allocator_, planar_) != TACV_SUCCESS)
    {
      error = "taCV hardware NV12 resize/CSC failed";
      return false;
    }
    return planar_.fd() >= 0 && planar_.mapped() != nullptr;
  }

  bool preprocess_bgr(std::string & error)
  {
    if (resized_bgr_.fd() < 0 || resized_bgr_.width() != config_.input_width ||
      resized_bgr_.height() != config_.input_height)
    {
      resized_bgr_ = topsbot_cv::TaImage::create(
        allocator_, FORMAT_BGR_PACKED, config_.input_width, config_.input_height);
    }
    if (planar_.fd() < 0 || planar_.width() != config_.input_width ||
      planar_.height() != config_.input_height || planar_.format() != FORMAT_BGR_PLANAR)
    {
      planar_ = topsbot_cv::TaImage::create(
        allocator_, FORMAT_BGR_PLANAR, config_.input_width, config_.input_height);
    }
    if (resized_bgr_.fd() < 0 || planar_.fd() < 0) {
      error = "failed to allocate taCV BGR intermediates";
      return false;
    }

    ta_cv_resize_t resize{};
    resize.in_width = source_bgr_.width();
    resize.in_height = source_bgr_.height();
    resize.out_width = config_.input_width;
    resize.out_height = config_.input_height;
    ta_cv_resize_image_t resize_image{};
    resize_image.resize_img_attr = &resize;
    resize_image.interpolation = interpolation_code(config_.interpolation);
    if (ta_cv_image_resize_hardware(
        &resize_image, source_bgr_.native(), resized_bgr_.native()) != TACV_SUCCESS ||
      ta_cv_image_bgrsplit(resized_bgr_.native(), planar_.native()) != TACV_SUCCESS)
    {
      error = "taCV hardware BGR resize/planar split failed";
      return false;
    }
    resized_bgr_.refresh_from_native(FORMAT_BGR_PACKED);
    planar_.refresh_from_native(FORMAT_BGR_PLANAR);
    return planar_.mapped() != nullptr;
  }

  void shutdown()
  {
    initialized_ = false;
    labels_.clear();
    source_nv12_.reset();
    jpeg_nv12_.reset();
    source_bgr_.reset();
    resized_bgr_.reset();
    planar_.reset();
    input_buffer_.reset();
    jpeg_input_buffer_.reset();
    if (output_created_ && context_ != 0) {
      (void)ta_runtime_destroy_buffer(&context_, &output_buffer_);
    }
    output_buffer_ = {};
    output_created_ = false;
    if (context_ != 0) {
      (void)ta_runtime_destroy_context(&context_);
      context_ = 0;
    }
    if (runtime_acquired_) {
      release_runtime();
      runtime_acquired_ = false;
    }
  }

  ModelConfig config_;
  topsbot_cv::DmabufAllocator allocator_;
  topsbot_cv::TaImage source_nv12_;
  topsbot_cv::TaImage jpeg_nv12_;
  topsbot_cv::TaImage source_bgr_;
  topsbot_cv::TaImage resized_bgr_;
  topsbot_cv::TaImage planar_;
  DmaBuffer input_buffer_;
  DmaBuffer jpeg_input_buffer_;
  ta_runtime_context context_{0};
  taconn_inout_attr_t input_attr_{};
  taconn_inout_attr_t output_attr_{};
  taconn_input_phy_t input_phy_{};
  taconn_buffer_t output_buffer_{};
  std::vector<std::string> labels_;
  bool runtime_acquired_{false};
  bool output_created_{false};
  bool initialized_{false};
};

}  // namespace topsbot_mobilenet_cls

PLUGINLIB_EXPORT_CLASS(
  topsbot_mobilenet_cls::MobileNetBackend, topsbot_ai_infer::IInferenceBackend)
