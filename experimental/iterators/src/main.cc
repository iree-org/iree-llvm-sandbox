#include <iostream>
#include <vector>

#include "operators/column_scan.h"
#include "utils/print.h"

int main(int /*unused*/, char ** /*unused*/) {
  std::vector<int32_t> Numbers = {1, 2, 3, 4};
  auto Scan = MakeColumnScanOperator(Numbers, Numbers, Numbers);
  Scan.open();
  decltype(Scan)::ReturnType Tuple;
  while ((Tuple = Scan.computeNext())) {
    PrintTuple(Tuple.value());
    std::cout << '\n';
  }
  Scan.close();
}
