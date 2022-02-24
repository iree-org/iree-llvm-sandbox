#include <gtest/gtest.h>

#include <vector>

#include "operators/column_scan.h"
#include "operators/reduce.h"

TEST(ReduceTest, SingleColumnSum) {
  std::vector<int32_t> Numbers = {1, 2, 3, 4};
  auto Scan = MakeColumnScanOperator(Numbers);
  auto Reduce = MakeReduceOperator(&Scan, [](auto T1, auto T2) {
    return std::make_tuple(std::get<0>(T1) + std::get<0>(T2));
  });
  Reduce.open();

  decltype(Scan)::ReturnType Tuple;

  // Consume one value
  Tuple = Reduce.computeNext();
  EXPECT_TRUE(Tuple);
  EXPECT_EQ(std::get<0>(Tuple.value()), 10);

  // Check that we have reached the end
  Tuple = Scan.computeNext();
  EXPECT_FALSE(Tuple);

  // Check that we can test for the end again
  Tuple = Scan.computeNext();
  EXPECT_FALSE(Tuple);

  Scan.close();
}

TEST(ReduceTest, MulticolumnMinMax) {
  std::vector<int32_t> Numbers = {1, 2, 3, 4};
  auto Scan = MakeColumnScanOperator(Numbers, Numbers);
  auto Reduce = MakeReduceOperator(&Scan, [](auto T1, auto T2) {
    return std::make_tuple(std::min(std::get<0>(T1), std::get<0>(T2)),
                           std::max(std::get<1>(T1), std::get<1>(T2)));
  });
  Reduce.open();

  decltype(Scan)::ReturnType Tuple;

  // Consume one value
  Tuple = Reduce.computeNext();
  EXPECT_TRUE(Tuple);
  EXPECT_EQ(std::get<0>(Tuple.value()), 1);
  EXPECT_EQ(std::get<1>(Tuple.value()), 4);

  // Check that we have reached the end
  Tuple = Scan.computeNext();
  EXPECT_FALSE(Tuple);

  Scan.close();
}
