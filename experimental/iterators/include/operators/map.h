//===-- operators/map.h - MapOperator definition ----------------*- C++ -*-===//
///
/// \file
/// This file contains the definition of the `MapOperator class` as well as
/// related helpers.
///
//===----------------------------------------------------------------------===//
#ifndef OPERATORS_MAP_H
#define OPERATORS_MAP_H

#include <optional>
#include <tuple>

/// Maps (or "transforms") all input tuples to a new one using a map function.
///
/// This operator consumes the tuples produced by its upstream operator and
/// transforms them using a given function.
template <typename UpstreamType, typename MapFunctionType>
class MapOperator {
public:
  using OutputTuple = decltype(std::declval<MapFunctionType>()(
      std::declval<typename UpstreamType::OutputTuple>()));
  using ReturnType = std::optional<OutputTuple>;

  /// Constructs a new `MapOperator` that holds a reference on its upstream
  /// operator and a copy of the provided `mapFunction`.
  explicit MapOperator(UpstreamType *const upstream,
                       MapFunctionType mapFunction)
      : upstream(upstream), mapFunction(std::move(mapFunction)) {}

  /// Opens the upstream operator.
  void open() { upstream->open(); }

  /// Consumes the tuples produced by the upstream operator and returns the
  /// result of applying the given mapFunction. Returns "end-of-stream" iff
  /// the upstream operator returns "end-of-stream".
  ReturnType computeNext() {
    const auto tuple = upstream->computeNext();

    if (!tuple) {
      return {};
    }

    return mapFunction(tuple.value());
  }

  /// Closes the upstream operator.
  void close() { upstream->close(); }

private:
  /// Reference to the upstream operator.
  UpstreamType *const upstream;
  /// Function that is used to transform the input tuples.
  MapFunctionType mapFunction;
};

/// Creates a new `MapOperator` deriving its template parameters from the
/// provided arguments.
template <typename UpstreamType, typename MapFunctionType>
auto makeMapOperator(UpstreamType *const upstream,
                     MapFunctionType mapFunction) {
  return MapOperator<UpstreamType, MapFunctionType>(upstream,
                                                    std::move(mapFunction));
}

#endif // OPERATORS_MAP_H
