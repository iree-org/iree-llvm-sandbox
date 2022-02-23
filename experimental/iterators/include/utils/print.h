#ifndef UTILS_PRINT_H_
#define UTILS_PRINT_H_

#include <iostream>
#include <tuple>

namespace impl {
template <typename TupleType, std::size_t... kIndices>
void PrintTupleTail(std::ostream &ostream, const TupleType &tuple,
                    const std::index_sequence<kIndices...> & /*unused*/) {
  ((ostream << ", " << std::get<kIndices + 1>(tuple)), ...);
}
}  // namespace impl

template <typename... Types>
void PrintTuple(std::ostream &ostream, const std::tuple<Types...> &tuple) {
  // Print first element
  ostream << "(" << std::get<0>(tuple);

  // Print remaining elements
  using TupleType = std::remove_reference_t<decltype(tuple)>;
  using IndexSequence =
      std::make_index_sequence<std::tuple_size_v<TupleType> - 1>;
  impl::PrintTupleTail(ostream, tuple, IndexSequence{});
  ostream << ")";
}

template <typename... Types>
void PrintTuple(const std::tuple<Types...> &tuple) {
    PrintTuple(std::cout, tuple);
}

#endif  // UTILS_PRINT_H_