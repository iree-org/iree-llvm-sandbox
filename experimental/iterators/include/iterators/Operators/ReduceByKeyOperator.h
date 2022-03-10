//===-- operators/reduce_by_key.h - ReduceByKeyOperator ---------*- C++ -*-===//
///
/// \file
/// This file contains the definition of the `ReduceByKeyOperator class` as
/// well as related helpers.
///
//===----------------------------------------------------------------------===//
#ifndef OPERATORS_REDUCE_BY_KEY_H
#define OPERATORS_REDUCE_BY_KEY_H

#include <optional>
#include <tuple>
#include <unordered_map>

#include "iterators/Utils/Tuple.h"

namespace mlir::iterators::operators {

/// Groups the input tuples by key and reduces the tuples within each group.
///
/// This operator consumes the tuples produced by its upstream operator, groups
/// them based on their key attributes (a given number of attributes starting
/// at the beginning of each tuple), and combines (aka reduces or folds) the
/// tuples of each group into a single tuple using the given reduce function.
/// If the upstream operator does not produce any tuple, neither does this
/// operator. The reduce function must accept two arguments that correspond
/// both to the value of the tuple produced by the upstream operator (i.e., the
/// remainder after the key attributes) and return a tuple of the same type.
template <typename UpstreamType, typename ReduceFunctionType,
          std::size_t kNumKeyAttributes>
class ReduceByKeyOperator {
public:
  using OutputTuple = typename UpstreamType::OutputTuple;
  using ReturnType = std::optional<OutputTuple>;

  /// Constructs a new `ReduceByKeyOperator` that holds a reference on its
  /// upstream operator and a copy of the provided `reduceFunction`.
  explicit ReduceByKeyOperator(UpstreamType *const upstream,
                               ReduceFunctionType reduceFunction)
      : upstream(upstream), reduceFunction(std::move(reduceFunction)),
        resultIt(result.begin()), resultEnd(result.end()) {}

  /// Opens the upstream operator.
  void open() { upstream->open(); }

  /// In the first call to this function, the tuples produced by the upstream
  /// operator are consumed and each tuple is combined with an intermediate
  /// internal aggregate of all previous tuples with the same key (i.e., in the
  /// same group). This first and the remaining calls return one fully reduced
  /// tuple per call. When no tuples remain in the result, "end-of-stream" is
  /// returned.
  ReturnType computeNext() {
    // Handle all corner cases: the first call, which triggers producing the
    // result in the state of this operator, and the last call, which signals
    // "end-of-stream".
    // Note that the two iterators have been initialized in the constructor
    // such that this tests succeeds in the first call. This allows to have
    // a single test in the common case.
    if (resultIt == resultEnd) {
      // All tuples have been returned; signal "end-of-stream"
      if (hasConsumedUpstream)
        return {};

      // This is the first call; we need to produce `result` from upstream
      consumeUpstream();
    }

    assert(hasConsumedUpstream);

    // We have a partially returned result, so return next element
    auto ret = std::tuple_cat(resultIt->first, resultIt->second);
    resultIt++;
    return ret;
  }

  /// Closes the upstream operator.
  void close() { upstream->close(); }

private:
  /// Runs the actual "reduce-by-key" logic while consuming all upstream.
  void consumeUpstream() {
    while (auto const tuple = upstream->computeNext()) {
      auto const key = utils::takeFront<kNumKeyAttributes>(tuple.value());
      auto const value = utils::dropFront<kNumKeyAttributes>(tuple.value());
      auto const [it, hasInserted] = result.emplace(key, value);
      if (!hasInserted)
        it->second = reduceFunction(it->second, value);
    }

    resultIt = result.begin();
    resultEnd = result.end();
    hasConsumedUpstream = true;
  }

  using KeyTuple = decltype(utils::takeFront<kNumKeyAttributes>(
      std::declval<OutputTuple>()));
  using ValueTuple = decltype(utils::dropFront<kNumKeyAttributes>(
      std::declval<OutputTuple>()));

  /// Reference to the upstream operator.
  UpstreamType *const upstream;
  /// Function used to reduce (or fold) upstream tuples pairwise.
  ReduceFunctionType reduceFunction;
  /// Holds the tuples to be returned by this operator (after the first call to
  /// `computeNext`).
  std::unordered_map<KeyTuple, ValueTuple, utils::TupleHasher<KeyTuple>> result;
  /// Iterator to the tuple returned by the next call to `computeNext`.
  typename decltype(result)::iterator resultIt;
  /// Past-the-end iterator of the tuples returned by this operator.
  typename decltype(result)::iterator resultEnd;
  /// Tracks whether this operator has consumed the tuples from upsteam (and
  /// thus whether `result` holds the tuples to be returned).
  bool hasConsumedUpstream = false;
};

/// Creates a new `ReduceByKeyOperator` deriving its template parameters from
/// the provided arguments.
template <std::size_t kNumKeyAttributes, typename UpstreamType,
          typename ReduceFunctionType>
auto makeReduceByKeyOperator(UpstreamType *const upstream,
                             ReduceFunctionType reduceFunction) {
  return ReduceByKeyOperator<UpstreamType, ReduceFunctionType,
                             kNumKeyAttributes>(upstream,
                                                std::move(reduceFunction));
}

} // namespace mlir::iterators::operators

#endif // OPERATORS_REDUCE_BY_KEY_H
