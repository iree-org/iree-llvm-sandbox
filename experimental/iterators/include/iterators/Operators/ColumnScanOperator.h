//===-- operators/column_scan.h - ColumnScanOperator definition -*- C++ -*-===//
///
/// \file
/// This file contains the definition of the `ColumnScanOperator class` as well
/// as related helpers.
///
//===----------------------------------------------------------------------===//
#ifndef ITERATORS_OPERATORS_COLUMNSCANOPERATOR_H
#define ITERATORS_OPERATORS_COLUMNSCANOPERATOR_H

#include <optional>
#include <tuple>
#include <vector>

namespace mlir::iterators::operators {

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
  explicit ColumnScanOperator(std::vector<InputTypes>... inputs)
      : inputs(std::make_tuple(std::move(inputs)...)), currentPos(0) {}

  /// Does nothing.
  void open() {}

  /// Returns a tuple consisting of the values in the vectors at the current
  /// index and increments that index if it is not past the end, or returns
  /// 'end of stream' otherwise.
  ReturnType computeNext() {
    // Signal end-of-stream if we are at the end of the input
    if (currentPos >= std::get<0>(inputs).size())
      return {};

    // Return current tuple and advance
    using IndexSequence =
        std::make_index_sequence<std::tuple_size_v<OutputTuple>>;
    auto const ret = extractCurrentTuple(IndexSequence{});
    currentPos++;
    return ret;
  }

  /// Does nothing.
  void close() {}

private:
  template <std::size_t... kIndices>
  std::optional<OutputTuple> extractCurrentTuple(
      const std::index_sequence<kIndices...> & /*unused*/) const {
    return std::make_tuple(std::get<kIndices>(inputs).at(currentPos)...);
  }

  /// Copy of the input data that this operator scans.
  std::tuple<std::vector<InputTypes>...> inputs;
  /// Position of the tuple values that are returned in the next call to
  /// `computeNext`.
  size_t currentPos;
};

/// Creates a new `ColumnScanOperator` deriving its template parameters from
/// the provided arguments.
template <typename... InputTypes>
auto makeColumnScanOperator(std::vector<InputTypes>... inputs) {
  return ColumnScanOperator<InputTypes...>(inputs...);
}

} // namespace mlir::iterators::operators

#endif // ITERATORS_OPERATORS_COLUMNSCANOPERATOR_H
