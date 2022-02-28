//===-- operators/reduce.h - ReduceOperator definition ----------*- C++ -*-===//
///
/// \file
/// This file contains the definition of the `ReduceOperator class` as well as
/// related helpers.
///
//===----------------------------------------------------------------------===//
#ifndef OPERATORS_REDUCE_H
#define OPERATORS_REDUCE_H

#include <optional>
#include <tuple>

/// Reduces (or "folds") all input tuples into one given a reduce function.
///
/// This operator consumes the tuples produced by its upstream operator and
/// combines (aka reduces or folds) them into a single tuple based on a given
/// reduce function. If the upstream does not produce any tuple, neither does
/// this operator. The reduce function must accept two arguments that are both
/// the same tuple as the upstream operator produces and return a tuple of the
/// same type.
template <typename UpstreamType, typename ReduceFunctionType>
class ReduceOperator {
public:
  using OutputTuple = typename UpstreamType::OutputTuple;
  using ReturnType = std::optional<OutputTuple>;

  /// Constructs a new `ReduceOperator` that holds a reference on its upstream
  /// operator and a copy of the provided `ReduceFunction`.
  explicit ReduceOperator(UpstreamType *const Upstream,
                          ReduceFunctionType ReduceFunction)
      : Upstream(Upstream), ReduceFunction(std::move(ReduceFunction)) {}

  /// Opens the upstream operator.
  void open() { Upstream->open(); }

  /// Consumes the tuples produced by the upstream operator and combines each
  /// with its internal aggregate to produce the new aggregate. Returns the
  /// final value of the aggregate if its upstream operator produced any
  /// output; returns "end-of-stream" otherwise.
  ReturnType computeNext() {
    // Consume and handle first tuple
    const auto FirstTuple = Upstream->computeNext();
    if (!FirstTuple) {
      return {};
    }

    // Aggregate remaining tuples
    OutputTuple Aggregate = FirstTuple.value();
    while (auto const Tuple = Upstream->computeNext()) {
      Aggregate = ReduceFunction(Aggregate, Tuple.value());
    }

    return Aggregate;
  }

  /// Closes the upstream operator.
  void close() { Upstream->close(); }

private:
  UpstreamType *const Upstream;
  ReduceFunctionType ReduceFunction;
};

/// Creates a new `ReduceOperator` deriving its template parameters from the
/// provided arguments.
template <typename UpstreamType, typename ReduceFunctionType>
auto makeReduceOperator(UpstreamType *const Upstream,
                        ReduceFunctionType ReduceFunction) {
  return ReduceOperator<UpstreamType, ReduceFunctionType>(
      Upstream, std::move(ReduceFunction));
}

#endif // OPERATORS_REDUCE_H
