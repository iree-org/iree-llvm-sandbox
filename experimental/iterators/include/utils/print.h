#ifndef UTILS_PRINT_H_
#define UTILS_PRINT_H_

#include <iostream>
#include <tuple>

namespace impl {
template <typename TupleType, std::size_t... kIndices>
void PrintTupleTail(const TupleType &tuple,
                    const std::index_sequence<kIndices...> & /*unused*/) {
  ((std::cout << ", " << std::get<kIndices>(tuple)), ...);
}
}  // namespace impl

template <typename... Types>
void PrintTuple(const std::tuple<Types...> &tuple) {
  // Print first element
  std::cout << "(" << std::get<0>(tuple);

  // Print remaining elements
  using TupleType = std::remove_reference_t<decltype(tuple)>;
  using IndexSequence =
      std::make_index_sequence<std::tuple_size_v<TupleType> - 1>;
  impl::PrintTupleTail(tuple, IndexSequence{});
  std::cout << ")";
}

#endif  // UTILS_PRINT_H_