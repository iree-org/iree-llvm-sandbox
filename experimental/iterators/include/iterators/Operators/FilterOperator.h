//===-- operators/filter.h - FilterOperator definition ----------*- C++ -*-===//
///
/// \file
/// This file contains the definition of the `FilterOperator class` as well as
/// related helpers.
///
//===----------------------------------------------------------------------===//
#ifndef OPERATORS_FILTER_H
#define OPERATORS_FILTER_H

#include <optional>
#include <tuple>

namespace mlir::iterators::operators {

/// Produces all tuples from upstream that pass the given filter.
///
/// This operator returns all those tuples produced by its upstream operator
/// for which the given filterFunction returns true. That function must thus
/// accept an argument of the tuple type returned by the upstream operator and
/// return bool.
template <typename UpstreamType, typename FilterFunctionType>
class FilterOperator {
public:
  using OutputTuple = typename UpstreamType::OutputTuple;
  using ReturnType = std::optional<OutputTuple>;

  /// Constructs a new `FilterOperator` that holds a reference on its upstream
  /// operator and a copy of the provided `filterFunction`.
  explicit FilterOperator(UpstreamType *const upstream,
                          FilterFunctionType filterFunction)
      : upstream(upstream), filterFunction(std::move(filterFunction)) {}

  /// Opens the upstream operator.
  void open() { upstream->open(); }

  /// Consumes tuples produced by the upstream operator, applies the given
  /// filter function, and returns the first tuple for which the filter returns
  /// true. Returns "end-of-stream" when upstream returns "end-of-stream".
  ReturnType computeNext() {
    while (auto const tuple = upstream->computeNext())
      if (filterFunction(tuple.value()))
        return tuple.value();

    return {};
  }

  /// Closes the upstream operator.
  void close() { upstream->close(); }

private:
  /// Reference to the upstream operator.
  UpstreamType *const upstream;
  /// Predicate function that determines which tuples pass the filter.
  FilterFunctionType filterFunction;
};

/// Creates a new `FilterOperator` deriving its template parameters from the
/// provided arguments.
template <typename UpstreamType, typename FilterFunctionType>
auto makeFilterOperator(UpstreamType *const upstream,
                        FilterFunctionType filterFunction) {
  return FilterOperator<UpstreamType, FilterFunctionType>(
      upstream, std::move(filterFunction));
}

} // namespace mlir::iterators::operators

#endif // OPERATORS_FILTER_H
