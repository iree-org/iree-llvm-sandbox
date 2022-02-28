# Stand-Alone Iterators Showcase

This folder contains a simple stand-alone project that implements
database-style iterators (aka open/next/close iterators or "Volcano" iterators)
in C++ using templates.

## Motivation

This code should serve as an illustration for how
[database-style iterators](https://db.in.tum.de/~grust/teaching/ws0607/MMDBMS/DBMS-CPU-5.pdf)
typically work in order to identify key concepts for implementing such
iterators in MLIR. Database systems use a tree of such iterators to represent
their executable query plans. Each iterator keeps its own state as well as
references to its children, and implements its execution logic, for which it
typically consumes its children's outputs. As far as I know, such stream-based
computation is not yet supported in existing MLIR dialects, but is key in
supporting data-intensive applications.

## Core Idea

This show-case implements database-style iterators in C++ using templates. This
allows to achieve the same composability of traditional database operators,
which use virtual interfaces, while keeping the overhead of that abstraction
to a mimimum. For that purpose, each operator class is a template that gets
parametrized with the exact type of its children (and potential other
parameters):

```cpp
template <typename UpstreamType, typename ReduceFunctionType>
class ReduceOperator {
public:
  using OutputTuple = typename UpstreamType::OutputTuple;
  using ReturnType = std::optional<OutputTuple>;
  // ...
  ReturnType computeNext() {
    // ...
    upstream->computeNext();  // Specialization is known, can be inlined
    // ...
```

Each operators returns `std::tuple`s in its `computeNext` function; the field
types depend on the query and are computed from the tuple types returned by the
children (and potentially other things) using templates. `computeNext` returns
an `std::optional`, where an empty optional indicates that the end of the
stream has been reached, i.e., no further tuples can be returned.

Each operator has a `Make*Operator` factory function that derives the template
parameters for the to-be-instatiated class such that assembling query plans is
concise:

```cpp
std::vector<int32_t> numbers = {1, 2, 3, 4};
auto scan = MakeColumnScanOperator(numbers);
auto reduce = MakeReduceOperator(&scan, [](auto t1, auto t2) {
  return std::make_tuple(std::get<0>(t1) + std::get<0>(t2));
});
```
