//===-- operators/column_scan.h - ColumnScanOperator definition -*- C++ -*-===//
///
/// \file
/// This file contains the definition of the `ColumnScanOperator class` as well
/// as related helpers.
///
//===----------------------------------------------------------------------===//
#ifndef OPERATORS_COLUMN_SCAN_H
#define OPERATORS_COLUMN_SCAN_H

#include <optional>
#include <tuple>
#include <vector>

/// Co-iterates over a set of vectors representing the columns of a table.
///
/// This operator takes as an input a set of vectors that represent the columns
/// of a table, i.e., a "struct of arrays". It co-iterates those vectors by
/// returning one row at the time, each consisting of the various values in the
/// vectors at the same index. This operator is thus a "source" operator.
template <typename... InputTypes>
class ColumnScanOperator {
public:
  using OutputTuple = std::tuple<InputTypes...>;
  using ReturnType = std::optional<OutputTuple>;

  /// Copies the given vectors into its internal state. All provided vectors
  /// have to have the same length.
  explicit ColumnScanOperator(std::vector<InputTypes>... Inputs)
      : Inputs(std::make_tuple(std::move(Inputs)...)), CurrentPos(0) {}

  /// Does nothing.
  void open() {}

  /// Returns a tuple consisting of the values in the vectors at the current
  /// index and increments that index if it is not past the end, or returns
  /// 'end of stream' otherwise.
  ReturnType computeNext() {
    // Signal end-of-stream if we are at the end of the input
    if (CurrentPos >= std::get<0>(Inputs).size()) {
      return {};
    }

    // Return current tuple and advance
    using IndexSequence =
        std::make_index_sequence<std::tuple_size_v<OutputTuple>>;
    auto const Ret = ExtractCurrentTuple(IndexSequence{});
    CurrentPos++;
    return Ret;
  }

  /// Does nothing.
  void close() {}

private:
  template <std::size_t... kIndices>
  std::optional<OutputTuple> ExtractCurrentTuple(
      const std::index_sequence<kIndices...> & /*unused*/) const {
    return std::make_tuple(std::get<kIndices>(Inputs).at(CurrentPos)...);
  }

  std::tuple<std::vector<InputTypes>...> Inputs;
  int64_t CurrentPos;
};

/// Creates a new `ColumnScanOperator` deriving its template parameters from
/// the provided arguments.
template <typename... InputTypes>
auto MakeColumnScanOperator(std::vector<InputTypes>... Inputs) {
  return ColumnScanOperator<InputTypes...>(Inputs...);
}

#endif // OPERATORS_COLUMN_SCAN_H
