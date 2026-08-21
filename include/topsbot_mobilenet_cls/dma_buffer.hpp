// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#ifndef TOPSBOT_MOBILENET_CLS__DMA_BUFFER_HPP_
#define TOPSBOT_MOBILENET_CLS__DMA_BUFFER_HPP_

#include <cstddef>
#include <cstdint>

namespace topsbot_cv
{
class DmabufAllocator;
}  // namespace topsbot_cv

namespace topsbot_mobilenet_cls
{

enum class CpuAccess
{
  Read,
  Write,
  ReadWrite
};

/// Issue DMA_BUF_IOCTL_SYNC around a CPU mmap access.
bool dmabuf_cpu_sync(int fd, CpuAccess access, bool start);

/// Resolve a physically contiguous DMA buffer for ta_runtime_set_input_pha.
bool get_physical_address(
  topsbot_cv::DmabufAllocator & allocator,
  int fd,
  uint64_t & address,
  size_t & length);

/// MobileNet-private movable RAII wrapper for a mapped DMA buffer.
class DmaBuffer
{
public:
  DmaBuffer() = default;
  ~DmaBuffer();

  DmaBuffer(DmaBuffer && other) noexcept;
  DmaBuffer & operator=(DmaBuffer && other) noexcept;

  DmaBuffer(const DmaBuffer &) = delete;
  DmaBuffer & operator=(const DmaBuffer &) = delete;

  bool allocate(topsbot_cv::DmabufAllocator & allocator, size_t size);
  bool allocate_at_least(topsbot_cv::DmabufAllocator & allocator, size_t minimum_size);
  void reset();

  bool begin_cpu_access(CpuAccess access);
  bool end_cpu_access();

  int fd() const {return fd_;}
  void * mapped() const {return mapped_;}
  size_t size() const {return size_;}
  bool valid() const {return fd_ >= 0 && mapped_ != nullptr;}

private:
  int fd_{-1};
  void * mapped_{nullptr};
  size_t size_{0};
  CpuAccess active_access_{CpuAccess::Read};
  bool cpu_access_active_{false};
};

}  // namespace topsbot_mobilenet_cls

#endif  // TOPSBOT_MOBILENET_CLS__DMA_BUFFER_HPP_
