//===-- Main.cpp - Example for stand-alone C++ iterators --------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <vector>

#include "iterators/Operators/ColumnScanOperator.h"
#include "iterators/Utils/Tuple.h"

using namespace database_iterators::operators;
using namespace database_iterators::utils;

int main(int /*unused*/, char ** /*unused*/) {
  std::vector<int32_t> numbers = {1, 2, 3, 4};
  auto scan = makeColumnScanOperator(numbers, numbers, numbers);
  scan.open();
  decltype(scan)::ReturnType tuple;
  while ((tuple = scan.computeNext())) {
    printTuple(tuple.value());
    std::cout << '\n';
  }
  scan.close();
}
