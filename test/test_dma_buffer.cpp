// Copyright 2026 TOPSBOT
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <utility>

#include <gtest/gtest.h>

#include "topsbot_mobilenet_cls/dma_buffer.hpp"
#include "topsbot_cv/dmabuf_allocator.hpp"

TEST(MobileNetDmaBuffer, AllocateSyncMoveAndReuse)
{
  topsbot_cv::DmabufAllocator allocator;
  if (!allocator.get()) {
    GTEST_SKIP() << "dmabuf heap is unavailable";
  }

  topsbot_mobilenet_cls::DmaBuffer source;
  if (!source.allocate(allocator, 4096U)) {
    GTEST_SKIP() << "carveout dmabuf allocation is unavailable";
  }
  ASSERT_TRUE(source.begin_cpu_access(topsbot_mobilenet_cls::CpuAccess::Write));
  std::memset(source.mapped(), 0x5A, source.size());
  ASSERT_TRUE(source.end_cpu_access());
  const int original_fd = source.fd();
  ASSERT_TRUE(source.allocate_at_least(allocator, 2048U));
  EXPECT_EQ(source.fd(), original_fd);

  topsbot_mobilenet_cls::DmaBuffer moved(std::move(source));
  EXPECT_FALSE(source.valid());
  ASSERT_TRUE(moved.valid());
  EXPECT_EQ(moved.fd(), original_fd);
  ASSERT_TRUE(moved.begin_cpu_access(topsbot_mobilenet_cls::CpuAccess::Read));
  EXPECT_EQ(static_cast<const unsigned char *>(moved.mapped())[2048], 0x5A);
  ASSERT_TRUE(moved.end_cpu_access());
  ASSERT_TRUE(moved.allocate_at_least(allocator, 8192U));
  EXPECT_TRUE(moved.valid());
  EXPECT_GE(moved.size(), 8192U);
}
