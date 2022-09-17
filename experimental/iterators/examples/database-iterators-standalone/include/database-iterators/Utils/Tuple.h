//===-- Tuple.h - Tuple-related utils ---------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_UTILS_TUPLE_H
#define ITERATORS_UTILS_TUPLE_H

#include <iostream>
#include <tuple>

namespace database_iterators::utils {

namespace impl {
template <typename TupleType, std::size_t... kIndices>
auto extractElements(const TupleType &tuple,
                     const std::index_sequence<kIndices...> & /*unused*/) {
  if constexpr (sizeof...(kIndices) > 0) {
    return std::make_tuple(std::get<kIndices>(tuple)...);
  } else {
    return std::make_tuple();
  }
}

template <std::size_t kFrontSize, typename TupleType, std::size_t... kIndices>
auto dropFront(const TupleType &tuple,
               const std::index_sequence<kIndices...> & /*unused*/) {
  using IndexSequence =
      std::integer_sequence<std::size_t, kIndices + kFrontSize...>;
  return extractElements(tuple, IndexSequence{});
}
} // namespace impl

/// Extracts the first `kFrontSize` elements from the given tuple.
template <std::size_t kFrontSize, typename... Types>
auto takeFront(const std::tuple<Types...> &tuple) {
  using IndexSequence = std::make_index_sequence<kFrontSize>;
  return impl::extractElements(tuple, IndexSequence{});
}

/// Extracts the last `kBackSize` elements from the given tuple.
template <std::size_t kFrontSize, typename... Types>
auto dropFront(const std::tuple<Types...> &tuple) {
  static constexpr std::size_t kBackSize = sizeof...(Types) - kFrontSize;
  using IndexSequence = std::make_index_sequence<kBackSize>;
  return impl::dropFront<kFrontSize>(tuple, IndexSequence{});
}

namespace impl {
template <std::size_t kIndex, typename TupleType>
std::size_t hashTupleField(const TupleType &tuple) {
  std::hash<std::tuple_element_t<kIndex, TupleType>> hasher;
  return hasher(std::get<kIndex>(tuple));
}

template <typename TupleType, std::size_t... kIndices>
std::size_t hashTuple(const TupleType &tuple,
                      const std::index_sequence<kIndices...> & /*unused*/) {
  return (hashTupleField<kIndices>(tuple) ^ ...);
}
} // namespace impl

/// Computes a hash value from an `std::tuple` combining the values of
/// `std::hash` of the individual fields using XOR.
template <typename... Types>
std::size_t hashTuple(const std::tuple<Types...> &tuple) {
  using IndexSequence = std::index_sequence_for<Types...>;
  return impl::hashTuple(tuple, IndexSequence{});
}

/// Functor class for `hashTuple`.
template <typename TupleType>
struct TupleHasher {
  std::size_t operator()(const TupleType &tuple) const {
    return hashTuple(tuple);
  }
};

namespace impl {
template <typename TupleType, std::size_t... kIndices>
void printTupleTail(std::ostream &ostream, const TupleType &tuple,
                    const std::index_sequence<kIndices...> & /*unused*/) {
  ((ostream << ", " << std::get<kIndices + 1>(tuple)), ...);
}
} // namespace impl

/// Prints the given tuple to the given stream using the format `"(a, b, c)"`.
template <typename... Types>
void printTuple(std::ostream &ostream, const std::tuple<Types...> &tuple) {
  // Print first element
  ostream << "(" << std::get<0>(tuple);

  // Print remaining elements
  using TupleType = std::remove_reference_t<decltype(tuple)>;
  using IndexSequence =
      std::make_index_sequence<std::tuple_size_v<TupleType> - 1>;
  impl::printTupleTail(ostream, tuple, IndexSequence{});
  ostream << ")";
}

/// Prints the given tuple to `std::cout` using the other overload.
template <typename... Types>
void printTuple(const std::tuple<Types...> &tuple) {
  printTuple(std::cout, tuple);
}

} // namespace database_iterators::utils

#endif // ITERATORS_UTILS_TUPLE_H
