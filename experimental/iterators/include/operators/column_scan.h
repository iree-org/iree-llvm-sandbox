#ifndef OPERATORS_COLUMN_SCAN_H
#define OPERATORS_COLUMN_SCAN_H

#include <optional>
#include <tuple>
#include <vector>

template <typename... InputTypes>
class ColumnScanOperator {
public:
  using OutputTuple = std::tuple<InputTypes...>;
  using ReturnType = std::optional<OutputTuple>;

  explicit ColumnScanOperator(std::vector<InputTypes>... Inputs)
      : Inputs(std::make_tuple(std::move(Inputs)...)), CurrentPos(0) {}

  void open() {}
  ReturnType computeNext() {
    // Signal end-of-stream if we are at the end of the input
    if (CurrentPos >= std::get<0>(Inputs).size()) {
      return {};
    }

    // Return current tuple and advance
    using IndexSequence =
        std::make_index_sequence<std::tuple_size_v<OutputTuple>>;
    auto const Ret = ExtractCurrentTuple(IndexSequence{});
    CurrentPos++;
    return Ret;
  }
  void close() {}

private:
  template <std::size_t... kIndices>
  std::optional<OutputTuple> ExtractCurrentTuple(
      const std::index_sequence<kIndices...> & /*unused*/) const {
    return std::make_tuple(std::get<kIndices>(Inputs).at(CurrentPos)...);
  }

  std::tuple<std::vector<InputTypes>...> Inputs;
  int64_t CurrentPos;
};

template <typename... InputTypes>
auto MakeColumnScanOperator(std::vector<InputTypes>... Inputs) {
  return ColumnScanOperator<InputTypes...>(Inputs...);
}

#endif // OPERATORS_COLUMN_SCAN_H
