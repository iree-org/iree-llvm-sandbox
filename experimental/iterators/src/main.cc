#include <iostream>
#include <vector>

#include "operators/column_scan.h"
#include "utils/tuple.h"

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
