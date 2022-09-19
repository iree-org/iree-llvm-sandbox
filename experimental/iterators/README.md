# Iterators dialect for MLIR

This folder contains an MLIR dialect based on the concepts of database-style
itertors, which allow to express computations on streams of data.

## Motivation

Database systems (both relational and non-relational) often implement
domain-specific compilers for high-level optimizations of their user programs
(or "queries") and, increasingly so, use JIT-compilers to reduce runtime
overheads. While this seems to make MLIR a good framework for implementing the
compilers of these systems, computations of their domains have some properties
that MLIR is currently unprepared for. This dialect is a first step to bridge
that gap: its goal is to implement database-style "iterators," which database
systems typically use in their execution layer.

## Background

### Analogy to Iterators in Python

Python's concept of [iterators](https://docs.python.org/3/glossary.html#term-iterator)
is very similar to that of the database-style iterators that this project is
targeting. The subsequent explains Python iterators in order to establish
fundamental concepts and terminology in a context that is familiar to many
readers.

Python's iterators allow to work with sequences of values without materializing
these sequences. This allows to (1) save memory for storing the entire sequence,
which may allow to do computations that would otherwise exceed the main memory,
(2) avoid computing values in the sequence that are never used, (3) as a
concequence, work with inifite sequences, and (4) improve cache efficiency
because the consumer of the values of a sequence is interleaved with the
producer of that sequence.

Consider the following example:

```python3
from itertools import *
for x in islice(count(10, 2), 3):
  print(x)
# Output:
# 10
# 12
# 14
```

Note that
[`count`](https://docs.python.org/3/library/itertools.html#itertools.count)
produces an infinite sequence -- it starts counting at the number given by first
argument (and the increment given as the second) -- but
[`islice`](https://docs.python.org/3/library/itertools.html#itertools.islice)
only consumes the first 3 values of it (as instructed to by its second
argument).

Formally, an [iterator](https://docs.python.org/3/glossary.html#term-iterator)
in Python is simply an object with a `__next__()` function, which, when called
repeatedly, return the values of the sequence defined by the iterator (and an
`__iter__()` function, which allows to use the iterator wherever an
[`iterable`](https://docs.python.org/3/glossary.html#term-iterable) is needed).
Consider the following class, which fulfills this requirement:

```python3
class CountToThree:
  def __init__(self):
    self.i = 0
  def __next__(self):
    if self.i == 3:
      raise StopIteration
    self.i += 1
    return self.i
  def __iter__(self):
    return self

for i in CountToThree():
  print(i)

# Output:
# 1
# 2
# 3
```

What is happening behind the scenes is that the `for` loop takes an instances of
the class `CountToThree`, calls `__iter__()` on that instance (which is required
to be an `iterable`) in order to obtain an `iterator`, and then calls
`__next__()` on that `iterator` until a `StopIteration` exceptions indicates the
end of the sequence, binding each resulting value to the loop variable. This
roughly corresponds to:

```python3
iterable = CountToThree()
iterator = iterable.__iter__()
print(iterator.__next__()) # Outputs: 1
print(iterator.__next__()) # Outputs: 2
print(iterator.__next__()) # Outputs: 3
print(iterator.__next__()) # Raises: StopIteration
```

One property that makes iterators in Python so powerful is that they are
composable. Many
[built-in functions](https://docs.python.org/3/library/functions.html) (such as
`enumerate`, `filter`, `map`, etc.) and standard library functions (most notably
those defined in
[`itertools`](https://docs.python.org/3/library/itertools.html)) accept
arguments of type `iterable` and produce again results that are `iterable`.
I/O is also often done with `iterable` objects (for example, via
[`IOBase`](https://docs.python.org/3/library/io.html#io.IOBase)).

Consider the following example, which computes the maximum length of a line with
an even number of characters of all lines in a given text file:

```python3
max(
  filter(lambda x: x % 2 == 0,
         map(lambda x: len(x),
             open('file.txt'))))
```

Each of the functions `max`, `filter`, `map`, and `open` either accepts or
produces an `iterator` object (or both). The result of the three inner-most
function is an `iterator` object that has two more `iterator` objects
recursively nested insise of them. That object is given to `max`, which drives
the computation by calling `__next__()` repeatably on that object, which, in
turn, causes a cascade of calls to `__next__()` of the nested `iterator`
objects. Since the state in each `iterator` object and the number of values
passed between them is constant, the above example essentially works on
arbitrarily large files.

### Iterators in Database Systems

Database systems (both relational and non-relational) typically express their
computations as transformations of streams of data in order to reduce data
movement between slow and fast memory (e.g., between DRAM and CPU cache).
Transformations are typically made composable through an "iterator" interface:
Each iterator produces a stream of data that can be consumed "downstream" one
element at a time, and, in turn, consumes the streams from its zero or more
"upstream" iterators to do so -- without knowing anything about its upstream and
downstream iterators. Each iterator may have complex control flow logic, often
depending on the data that it consumes, and can manage its own state. The
iterator interface, thus, passes data *and* control: when one iterator asks its
upstream iterator for the next element, it passes control to that iterator,
which returns the control together with the next element. By limiting the number
of in-flight elements to essentially one, this minimizes the overall state and
thus data movement.

TODO(ingomueller): explain Volcano iterator interface
