//===-- HashJoinOperator.h - HashJoinOperator -------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the `HashJoinOperator class` as well
/// as related helpers.
///
//===----------------------------------------------------------------------===//
#ifndef ITERATORS_OPERATORS_HASHJOINOPERATOR_H
#define ITERATORS_OPERATORS_HASHJOINOPERATOR_H

#include <optional>
#include <tuple>
#include <unordered_map>

#include "database-iterators/Utils/Tuple.h"

namespace database_iterators::operators {

/// Returns tuples from its two upstream operators that have matching keys.
///
/// This operator consumes the tuples produced by its two upstream operators
/// and returns any combination of tuples that have matching keys, i.e., where
/// the first given number of attributes compare equal. The returned tuple
/// consist of the key attributes, the remaining attributes from the left side,
/// and the remaining attributes from the right side. The operators implements
/// a hash join: It first builds a hash table from the left input, which is
/// called "build" side; then, while consuming the right input, which is called
/// "probe" side, probes the hash table for matching tuples in order to produce
/// the next output tuple.
template <typename BuildUpstreamType, typename ProbeUpstreamType,
          std::size_t kNumKeyAttributes>
class HashJoinOperator {
  using BuildSideInputTuple = typename BuildUpstreamType::OutputTuple;
  using ProbeSideInputTuple = typename ProbeUpstreamType::OutputTuple;

  using KeyTuple = decltype(utils::takeFront<kNumKeyAttributes>(
      std::declval<BuildSideInputTuple>()));
  using BuildSideValueTuple = decltype(utils::dropFront<kNumKeyAttributes>(
      std::declval<BuildSideInputTuple>()));
  using ProbeSideValueTuple = decltype(utils::dropFront<kNumKeyAttributes>(
      std::declval<ProbeSideInputTuple>()));
  using ValueTuple =
      decltype(std::tuple_cat(std::declval<BuildSideValueTuple>(),
                              std::declval<ProbeSideValueTuple>()));

public:
  using OutputTuple = decltype(std::tuple_cat(std::declval<KeyTuple>(),
                                              std::declval<ValueTuple>()));
  using ReturnType = std::optional<OutputTuple>;

  /// Constructs a new `HashJoinOperator` that holds a reference on its
  /// upstream operators.
  explicit HashJoinOperator(BuildUpstreamType *const buildUpstream,
                            ProbeUpstreamType *const probeUpstream)
      : buildUpstream(buildUpstream), probeUpstream(probeUpstream),
        currentBuildSideMatchesIt(buildTable.end()),
        currentBuildSideMatchesEnd(buildTable.end()) {}

  /// Builds the hash table from all output of the build-side upstream
  /// operator and opens the probe-side upstream operator.
  void open() {
    buildUpstream->open();
    while (auto const tuple = buildUpstream->computeNext()) {
      auto const key = utils::takeFront<kNumKeyAttributes>(tuple.value());
      auto const value = utils::dropFront<kNumKeyAttributes>(tuple.value());
      buildTable.emplace(key, value);
    }
    buildUpstream->close();

    probeUpstream->open();
  }

  /// Returns the next matching tuple. Each tuple from the probe side may have
  /// several matches from the build side in the hash table, so this function
  /// iterates over matches of some previous probe-side tuple in the general
  /// case. When no more match exists (and in the first call), the function
  /// consumes tuples from the probe side until it finds a match and
  /// initializes the iteration logic with that match. Returns "end-of-stream"
  /// when the last match of the last probe-side tuple has been returned.
  ReturnType computeNext() {
    // Consume from probe-side upstream until we find a match in the build table
    while (currentBuildSideMatchesIt == currentBuildSideMatchesEnd) {
      auto const currentProbeTuple = probeUpstream->computeNext();

      // No more input from the probe-side side, so we can't produce any more
      /// output
      if (!currentProbeTuple)
        return {};

      // Look up key of current tuple from the probe-side in the build table.
      auto const key =
          utils::takeFront<kNumKeyAttributes>(currentProbeTuple.value());
      currentProbeSideValue =
          utils::dropFront<kNumKeyAttributes>(currentProbeTuple.value());
      std::tie(currentBuildSideMatchesIt, currentBuildSideMatchesEnd) =
          buildTable.equal_range(key);
    }

    // Return a combination of the build-side tuple and the next matching tuple
    // from the probe side
    auto ret = std::tuple_cat(currentBuildSideMatchesIt->first,
                              currentBuildSideMatchesIt->second,
                              currentProbeSideValue);
    currentBuildSideMatchesIt++;
    return ret;
  }

  /// Closes the probe-side upstream operator.
  void close() { probeUpstream->close(); }

private:
  /// Reference to the left (build-side) upstream operator.
  BuildUpstreamType *const buildUpstream;
  /// Reference to the right (probe-side) upstream operator.
  ProbeUpstreamType *const probeUpstream;
  /// Hash table containing all tuples from the build side (after `open` has
  /// been called).
  std::unordered_multimap<KeyTuple, BuildSideValueTuple,
                          utils::TupleHasher<KeyTuple>>
      buildTable;
  /// Value (i.e., tuple of non-key attributes) from the last probe-side tuple
  /// that will be part of the tuples returned for matching build-side tuples.
  ProbeSideValueTuple currentProbeSideValue;
  /// Iterator to the next element in `buildTable` that matches the last
  /// probe-side tuple.
  typename decltype(buildTable)::iterator currentBuildSideMatchesIt;
  /// Past-the-end iterator of elements in `buildTable` that match the last
  /// probe-side tuple.
  typename decltype(buildTable)::iterator currentBuildSideMatchesEnd;
};

/// Creates a new `HashJoinOperator` deriving its template parameters from the
/// provided arguments.
template <std::size_t kNumKeyAttributes, typename BuildUpstreamType,
          typename ProbeUpstreamType>
auto makeHashJoinOperator(BuildUpstreamType *const buildUpstream,
                          ProbeUpstreamType *const probeUpstream) {
  return HashJoinOperator<BuildUpstreamType, ProbeUpstreamType,
                          kNumKeyAttributes>(buildUpstream, probeUpstream);
}

} // namespace database_iterators::operators

#endif // ITERATORS_OPERATORS_HASHJOINOPERATOR_H
