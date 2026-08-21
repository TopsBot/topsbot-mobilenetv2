// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#include "topsbot_mobilenet_cls/dma_buffer.hpp"

#include <cerrno>
#include <utility>

#include <linux/dma-buf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <BufferAllocator/BufferAllocatorWrapper.h>

#include "topsbot_cv/dmabuf_allocator.hpp"

namespace topsbot_mobilenet_cls
{

namespace
{
uint64_t access_flags(const CpuAccess access)
{
  switch (access) {
    case CpuAccess::Read:
      return DMA_BUF_SYNC_READ;
    case CpuAccess::Write:
      return DMA_BUF_SYNC_WRITE;
    case CpuAccess::ReadWrite:
      return DMA_BUF_SYNC_RW;
  }
  return DMA_BUF_SYNC_RW;
}
}  // namespace

bool dmabuf_cpu_sync(const int fd, const CpuAccess access, const bool start)
{
  if (fd < 0) {
    return false;
  }
  dma_buf_sync sync{};
  sync.flags = access_flags(access) | (start ? DMA_BUF_SYNC_START : DMA_BUF_SYNC_END);
  int result = -1;
  do {
    result = ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
  } while (result < 0 && errno == EINTR);
  return result == 0;
}

bool get_physical_address(
  topsbot_cv::DmabufAllocator & allocator,
  const int fd,
  uint64_t & address,
  size_t & length)
{
  address = 0;
  length = 0;
  BufferAllocator * const raw_allocator = allocator.get();
  if (fd < 0 || raw_allocator == nullptr) {
    return false;
  }

  uint64_t raw_length = 0;
  if (DmabufHeapGetPhysAddr(
      raw_allocator, static_cast<unsigned int>(fd), &address, &raw_length) < 0)
  {
    address = 0;
    return false;
  }
  length = static_cast<size_t>(raw_length);
  return address != 0 && length != 0;
}

DmaBuffer::~DmaBuffer()
{
  reset();
}

DmaBuffer::DmaBuffer(DmaBuffer && other) noexcept
: fd_(other.fd_),
  mapped_(other.mapped_),
  size_(other.size_),
  active_access_(other.active_access_),
  cpu_access_active_(other.cpu_access_active_)
{
  other.fd_ = -1;
  other.mapped_ = nullptr;
  other.size_ = 0;
  other.cpu_access_active_ = false;
}

DmaBuffer & DmaBuffer::operator=(DmaBuffer && other) noexcept
{
  if (this != &other) {
    reset();
    fd_ = other.fd_;
    mapped_ = other.mapped_;
    size_ = other.size_;
    active_access_ = other.active_access_;
    cpu_access_active_ = other.cpu_access_active_;
    other.fd_ = -1;
    other.mapped_ = nullptr;
    other.size_ = 0;
    other.cpu_access_active_ = false;
  }
  return *this;
}

bool DmaBuffer::allocate(
  topsbot_cv::DmabufAllocator & allocator,
  const size_t size)
{
  if (valid() && size_ == size) {
    return true;
  }
  reset();
  if (size == 0) {
    return false;
  }

  fd_ = allocator.alloc(size);
  if (fd_ < 0) {
    return false;
  }
  mapped_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (mapped_ == MAP_FAILED) {
    mapped_ = nullptr;
    close(fd_);
    fd_ = -1;
    return false;
  }
  size_ = size;
  return true;
}

bool DmaBuffer::allocate_at_least(
  topsbot_cv::DmabufAllocator & allocator,
  const size_t minimum_size)
{
  if (valid() && size_ >= minimum_size) {
    return true;
  }
  return allocate(allocator, minimum_size);
}

void DmaBuffer::reset()
{
  if (cpu_access_active_) {
    (void)end_cpu_access();
  }
  if (mapped_ != nullptr && size_ != 0) {
    munmap(mapped_, size_);
  }
  if (fd_ >= 0) {
    close(fd_);
  }
  fd_ = -1;
  mapped_ = nullptr;
  size_ = 0;
}

bool DmaBuffer::begin_cpu_access(const CpuAccess access)
{
  if (!valid() || cpu_access_active_) {
    return false;
  }
  if (!dmabuf_cpu_sync(fd_, access, true)) {
    return false;
  }
  active_access_ = access;
  cpu_access_active_ = true;
  return true;
}

bool DmaBuffer::end_cpu_access()
{
  if (!cpu_access_active_) {
    return false;
  }
  const bool result = dmabuf_cpu_sync(fd_, active_access_, false);
  cpu_access_active_ = false;
  return result;
}

}  // namespace topsbot_mobilenet_cls
