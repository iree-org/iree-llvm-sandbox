#ifndef OPERATORS_REDUCE_H
#define OPERATORS_REDUCE_H

#include <optional>
#include <tuple>

template <typename UpstreamType, typename ReduceFunctionType>
class ReduceOperator {
public:
  using OutputTuple = typename UpstreamType::OutputTuple;
  using ReturnType = std::optional<OutputTuple>;

  explicit ReduceOperator(UpstreamType *const Upstream,
                          ReduceFunctionType ReduceFunction)
      : Upstream(Upstream), ReduceFunction(std::move(ReduceFunction)) {}

  void open() { Upstream->open(); }
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
  void close() { Upstream->close(); }

private:
  UpstreamType *const Upstream;
  ReduceFunctionType ReduceFunction;
};

template <typename UpstreamType, typename ReduceFunctionType>
auto MakeReduceOperator(UpstreamType *const Upstream,
                        ReduceFunctionType ReduceFunction) {
  return ReduceOperator<UpstreamType, ReduceFunctionType>(
      Upstream, std::move(ReduceFunction));
}

#endif // OPERATORS_REDUCE_H
