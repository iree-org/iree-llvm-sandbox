#include <iostream>
#include <vector>

#include "operators/column_scan.h"
#include "utils/print.h"

int main(int, char **) {
  std::vector<int32_t> numbers = {1, 2, 3, 4};
  ColumnScanOperator<int32_t, int32_t, int32_t> scan(numbers, numbers, numbers);
  decltype(scan)::ReturnType tuple;
  while ((tuple = scan.ComputeNext())) {
    PrintTuple(tuple.value());
    std::cout << std::endl;
  }
}
