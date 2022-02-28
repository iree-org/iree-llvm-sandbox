#ifndef UTILS_PRINT_H_
#define UTILS_PRINT_H_

#include <iostream>
#include <tuple>

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

#endif // UTILS_PRINT_H_
