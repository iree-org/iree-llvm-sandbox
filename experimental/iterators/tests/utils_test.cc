#include <gtest/gtest.h>

#include <sstream>
#include <tuple>

#include "utils/print.h"

TEST(PrintTupleTest, SingleField) {
    std::stringstream string_buffer;
    PrintTuple(string_buffer, std::make_tuple(1));
    EXPECT_EQ(string_buffer.str(), "(1)");
}

TEST(PrintTupleTest, MultipleFields) {
    std::stringstream string_buffer;
    PrintTuple(string_buffer, std::make_tuple(1, 2, 3));
    EXPECT_EQ(string_buffer.str(), "(1, 2, 3)");
}