//===-- UtilsTest.cpp - Unit tests of utils ---------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include <sstream>
#include <tuple>

#include "iterators/Utils/Tuple.h"

using namespace mlir::iterators::utils;

TEST(ExtractHeadTest, Head0) {
  std::tuple<int16_t, int32_t> tuple = {1, 2};
  auto const head = takeFront<0>(tuple);
  static_assert(std::is_same_v<std::remove_cv_t<decltype(head)>, std::tuple<>>,
                "Expected head to be of type std::tuple<>.");
  EXPECT_EQ(head, std::make_tuple());
}

TEST(ExtractHeadTest, Head1) {
  std::tuple<int16_t, int32_t> tuple = {1, 2};
  auto const head = takeFront<1>(tuple);
  static_assert(
      std::is_same_v<std::remove_cv_t<decltype(head)>, std::tuple<int16_t>>,
      "Expected head to be of type std::tuple<int16_t>.");
  EXPECT_EQ(head, std::make_tuple(1));
}

TEST(ExtractHeadTest, Head2) {
  std::tuple<int16_t, int32_t> tuple = {1, 2};
  auto const head = takeFront<2>(tuple);
  static_assert(std::is_same_v<std::remove_cv_t<decltype(head)>,
                               std::tuple<int16_t, int32_t>>,
                "Expected head to be of type std::tuple<int16_t, int32_t>.");
  EXPECT_EQ(head, std::make_tuple(1, 2));
}

TEST(ExtractTailTest, Tail0) {
  std::tuple<int16_t, int32_t> tuple = {1, 2};
  auto const tail = dropFront<0>(tuple);
  static_assert(std::is_same_v<std::remove_cv_t<decltype(tail)>,
                               std::tuple<int16_t, int32_t>>,
                "Expected tail to be of type std::tuple<int16_t, int32_t>.");
  EXPECT_EQ(tail, std::make_tuple(1, 2));
}

TEST(ExtractTailTest, Tail1) {
  std::tuple<int16_t, int32_t> tuple = {1, 2};
  auto const tail = dropFront<1>(tuple);
  static_assert(
      std::is_same_v<std::remove_cv_t<decltype(tail)>, std::tuple<int32_t>>,
      "Expected tail to be of type std::tuple<int32_t>.");
  EXPECT_EQ(tail, std::make_tuple(2));
}

TEST(ExtractTailTest, Tail2) {
  std::tuple<int16_t, int32_t> tuple = {1, 2};
  auto const tail = dropFront<2>(tuple);
  static_assert(std::is_same_v<std::remove_cv_t<decltype(tail)>, std::tuple<>>,
                "Expected tail to be of type std::tuple<>.");
  EXPECT_EQ(tail, std::make_tuple());
}

TEST(HashTupleTest, SimpleTests) {
  std::hash<uint32_t> hasher;
  EXPECT_EQ(hashTuple(std::make_tuple(1)), hasher(1));
  EXPECT_EQ(hashTuple(std::make_tuple(1, 2)), hasher(1) ^ hasher(2));
}

TEST(PrintTupleTest, SingleField) {
  std::stringstream stringBuffer;
  printTuple(stringBuffer, std::make_tuple(1));
  EXPECT_EQ(stringBuffer.str(), "(1)");
}

TEST(PrintTupleTest, MultipleFields) {
  std::stringstream stringBuffer;
  printTuple(stringBuffer, std::make_tuple(1, 2, 3));
  EXPECT_EQ(stringBuffer.str(), "(1, 2, 3)");
}
