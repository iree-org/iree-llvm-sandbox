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

  explicit ColumnScanOperator(std::vector<InputTypes>... inputs)
      : inputs_(std::make_tuple(std::move(inputs)...)), current_pos_(0) {}

  void Open() {}
  ReturnType ComputeNext() {
    // Signal end-of-stream if we are at the end of the input
    if (current_pos_ >= std::get<0>(inputs_).size()) {
      return {};
    }

    // Return current tuple and advance
    using IndexSequence =
        std::make_index_sequence<std::tuple_size_v<OutputTuple>>;
    auto const ret = ExtractCurrentTuple(IndexSequence{});
    current_pos_++;
    return ret;
  }
  void Close() {}

 private:
  template <std::size_t... kIndices>
  std::optional<OutputTuple> ExtractCurrentTuple(
      const std::index_sequence<kIndices...> & /*unused*/) const {
    return std::make_tuple(std::get<kIndices>(inputs_).at(current_pos_)...);
  }

  std::tuple<std::vector<InputTypes>...> inputs_;
  int64_t current_pos_;
};

#endif  // OPERATORS_COLUMN_SCAN_H