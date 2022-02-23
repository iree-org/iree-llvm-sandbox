#ifndef OPERATORS_REDUCE_H
#define OPERATORS_REDUCE_H

#include <optional>
#include <tuple>

template <typename Upstream, typename ReduceFunction>
class ReduceOperator {
 public:
  using OutputTuple = typename Upstream::OutputTuple;
  using ReturnType = std::optional<OutputTuple>;

  explicit ReduceOperator(Upstream *const upstream,
                          ReduceFunction reduce_function)
      : upstream_(upstream), reduce_function_(std::move(reduce_function)) {}

  void Open() { upstream_->Open(); }
  ReturnType ComputeNext() {
    // Consume and handle first tuple
    const auto first_tuple = upstream_->ComputeNext();
    if (!first_tuple) {
      return {};
    }

    // Aggregate remaining tuples
    OutputTuple aggregate = first_tuple.value();
    while (auto const tuple = upstream_->ComputeNext()) {
      aggregate = reduce_function_(aggregate, tuple.value());
    }

    return aggregate;
  }
  void Close() { upstream_->Close(); }

 private:
  Upstream *const upstream_;
  ReduceFunction reduce_function_;
};

template <typename Upstream, typename ReduceFunction>
auto MakeReduceOperator(Upstream *const upstream,
                        ReduceFunction reduce_function) {
  return ReduceOperator<Upstream, ReduceFunction>(upstream,
                                                  std::move(reduce_function));
}

#endif  // OPERATORS_REDUCE_H