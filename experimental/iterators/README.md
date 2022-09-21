# Iterators dialect for MLIR

This folder contains an MLIR dialect based on the concepts of database-style
iterators, which allow to express computations on streams of data.

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

#### Motivation

Python's iterators allow to work with sequences of values without materializing
these sequences. This allows to (1) save memory for storing the entire sequence,
which may allow to do computations that would otherwise exceed the main memory,
(2) avoid computing values in the sequence that are never used, (3) as a
consequence, work with infinite sequences, and (4) improve cache efficiency
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

#### Definition of `iterator`

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

Note that there is only one iteration per `iterator`: if the `iterator` object
is passed around and `__next__()` is called on that object in different places,
each of them advances *the same iteration*. Doing so is, thus, most often a bad
idea; it is rather useful to limit each `iterator` to one consumer.

#### Composability

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
functions is an `iterator` object that has two more `iterator` objects
recursively nested inside of it. That object is given to `max`, which drives the
computation by calling `__next__()` repeatably on that object, which, in turn,
causes a cascade of calls to `__next__()` of the nested `iterator` objects.
Since the state in each `iterator` object and the number of values passed
between them is constant, the above example essentially works on arbitrarily
large files.

#### Distinction to Related Concepts in Python

A few related but distinct concepts deserve a short discussion:

* `iterable`: As mentioned before, `iterable`s in Python are things than can be
  iterated over (such as containers like `list`s, `tuple`s, etc) while
  `iterator`s are objects that help with that iteration. `iterable`s provide
  `iterator`s via their `__iter__()` function. The `for` loop uses that function
  to get an `iterator`, and uses that in turn to iterate. Confusion may arise
  because `iterator`s *are* also `iterable` (in order to make `iterator`s usable
  in `for` loops).
* `generator`: This term is actually ambiguous; it may refer to
  [`generator function`](https://docs.python.org/3/glossary.html#term-generator)
  (which seems to be the more common usage of simply `generator`) or
  [`generator iterator`](https://docs.python.org/3/glossary.html#term-generator-iterator).
  In short, the former is a function that returns the latter.
  In more detail, a `generator function` is a function with a `yield` statement,
  which, therefore, returns an object of the built-in type `generator` -- a
  class that implements the `iterator` interface. Similarly, a
  [`generator expression`](https://docs.python.org/3/glossary.html#term-generator-expression)
  (such as `i*i for i in range(10)`) is a language construct that, like
  `generator function`s, returns an object of the built-in type `generator`
  (which is an iterator).

### Iterators in Database Systems

Database systems (both relational and non-relational) typically express their
computations as transformations of streams of data in order to reduce data
movement between slow and fast memory (e.g., between DRAM and CPU cache). They
typically do so with "iterators" that are quite similar to those in Python, and
pretty much for the same reasons. First, having all transformations implement
some iterator interface makes them composable: Each iterator produces a stream
of data that can be consumed "downstream" one element at a time, and, in turn,
consumes the streams from its zero or more "upstream" iterators to do so --
without knowing anything about its upstream and downstream iterators. This helps
isolating the control flow of each iterator, which may be complex, as well as
its state. Like in Python, the iterator interface also allows for interleaved
execution of many different computation steps by passing data *and* control:
when one iterator asks its upstream iterator for the next element, it passes
control to that iterator, which returns the control together with the next
element. By limiting the number of in-flight elements to essentially one, this
minimizes the overall state and thus data movement.

Iterators in database systems are often implemented with some variant of the
so-called "Volcano" iterator model (named after the system that first proposed
this model, see [Graefe'90](https://doi.org/10.1145/93605.98720)). Conceptually,
the model is the same as those of Python's iterators; however, the communication
protocol between iterators is slightly different. Concretely, it defines that
each iterator class should have the functions `open`, `next`, and `close` (and
is therefore sometimes called "open/next/close" interface). Computations are
expressed as a tree of such iterators. (The reason why it has to be a tree is
that, like in Python, open/next/close iterators only provide one iteration, i.e.,
they can only be consumed in one place. In a tree, this place is their parent
iterator.) The computation is initialized by calling `open` on the root
iterator, which typically calls `open` on its child iterators, such that the
whole tree is initialized recursively. The computation is then driven by
calling `next` repeatedly on the root, each call producing one element of the
result until all of them have been produced. Each call to `next` on the root
typically triggers a cascade of `next` calls in the tree (but how many calls
are required in each of the iterators depends on those iterators, their current
state, and the input they are consuming). After a call to `next` has signaled
the end of the result stream, or when the system or user wants to abort the
computation, the `close` function of the root iterator is called, which
recursively calls `close` on its children, all of which clean up any state they
may have.

The [`examples/database-iterators-standalone`](examples/database-iterators-standalone)
folder contains a standalone project that implements the open/next/close
interface using C++ templates for illustration purposes.

Originally, the open/next/close interface was implemented with (virtual)
function calls in order to achieve the desired composability but this has
changed over time. In the times when almost all data managed by the database
system was stored on disk, the overhead of virtual function calls in the inner
loop was negligible. However, DRAM sizes have increased faster than typical
dataset sizes, so most data today is processed from main memory, in which case
virtual functions are not practical anymore. Several solutions have been
proposed over time, two of which are related variants of the Volcano model: (1)
Rather than passing one element at a time, it is possible to pass *batches* of
them in each call to `next`, and then express the computations as virtual
functions *on the whole batch*. This amortizes the virtual function calls with
the batch size but keeps the benefits of being cache-efficient. See the seminal
work on [MonetDB/X100](https://www.cidrdb.org/cidr2005/papers/P19.pdf) for
details. (2) Alternatively, JIT compilation (of the entire computation or parts
of it) has become widespread for eliminating the function call overhead,
as popularized by [HyPer](https://doi.org/10.14778/2002938.2002940). To some
degree, the [C++ template project](examples/database-iterators-standalone)
mentioned above achieves this by relying on the C++ compiler to inline templated
function calls; another widespread technique is to produce LLVM IR one way or
the other.

### The Relational Data Model

We briefly review the data model of relational database systems. To a large
degree, iterators seem orthogonal to the elements they iterate over but whether
or not and how we should achieve this orthogonality may require some discussion,
so reviewing the data model here may be useful.

Relational database systems manage *relations*. A relation is a bag (i.e.,
multiset) of *tuples* (or *records*). (Textbooks often define relations as
*sets* but practical systems use bag-based relational algebra, or even
sequence-based or mixed algebras in order to support order-sensitive operations
such as window functions.) A record is a collection of named and typed
*attributes* (or *fields* or, depending on the context, *columns*). Attributes
may be *nullable* (read: optional), i.e., they may contain a value of a
particular type or the special value *null* indicating the absence of a value.
In the plain relational model, attributes are atomic values (numbers, strings,
etc); in nested relational algebra, an attribute may be of a relation type or a
tuple type.

The closest equivalent to a relation in Python is arguably a
[`List[TupleType]`](https://docs.python.org/3/library/typing.html#typing.List),
where `TupleType` is some type based on
[`NamedTuple`](https://docs.python.org/3/library/typing.html#typing.NamedTuple),
e.g., `NamedTuple('Employee', [('name', Optional[str]), ('id', int)])`. One
inaccuracy in this analogy is the fact that (bag-based) relations do not specify
an order of their tuples while `List` does. Another inaccuracy is the fact that
`NamedTuple` specifies a name for the tuple type whereas this isn't (always) the
case in the relational model. Also, `NamedTuple` allows ordered access to the
fields whereas the field order does not play any role in the relational model
(though it does play a role in SQL, most notably for `UNION` and similar).

Another way to look at relations is to think of them as *tables*: each tuple
represents one row of the table and the corresponding attribute values of
different rows represent a column. The values in each column have the same type.
Again, this analogy is inaccurate in that it implies an order of the rows and
columns.

Relations, thus, have a certain similarity to matrices: they are also
two-dimensional and can be thought of in terms of rows and columns. However,
(1) only columns have values of the same type, whereas the values of one row
generally don't, (2) at least on the highest-level of abstraction (and modulo
the sequence-based relational model), neither the order of the rows nor that of
the columns matter, (3) the elements of relations may be strings or similar
variable-length data types, and (4) nested relational algebra also allows for
structured element types (which may have variable length).

## Basic Concepts

This section defines the central concepts of this dialect and relates them to
those introduced in the [Background](#background) section.

* **Stream**: A collection of elements of a particular type that (1) is ordered
  and (2) can only be iterated over in that order one element at the time.

  Stream is the main data type that iterators consume and produce.

  Ideally, there should be no restriction on the type of the elements and it
  seems like it is possible to achieve that ideal (since Python achieves it).

  We currently assume that streams are finite but most concepts will probably
  remain unchanged if that restriction were lifted (see Python supporting
  infinite iterators).

* **Iterator Op**: An operation producing a stream and consuming zero or more
  other streams (plus zero or more non-stream operands) identified by a name.

  An iterator op only specifies *that* streams are handled in a particular way
  but *how* that is happening is only specified in its documentation, not in the
  form of code.

  The equivalent to iterator ops in Python are functions returning iterators
  (like `map` and similar built-in functions as well as those from `itertools`),
  which are really "iterator factories" -- not iterators. We choose, say, `map`
  by name, which produces an iterator that behaves according to the
  documentation of `map`. We do *not* write down ourselves the imperative or
  comprehension-based logic of the iterator, *nor* do we see or need to
  understand the `iterator` interface.

  Iterator ops can be lowered to an implementation based on the Volcano iterator
  model but other lowerings are conceivable. (For example,
  [this project](https://github.com/Dinistro/circt-stream) lowers ops that are
  approximately equivalent to iterator ops to FPGAs.) Lowering materializes the
  semantics specified by the name of the iterator op in a lower-level form;
  similarly, the call to `map` materializes its semantics as an `iterator`
  object that behaves according its specification.

* **Iterator Body**: The function bodies of the functions `open`, `next`, and
  `close`.

  This is the equivalent of Python's `iterator`s, though with a different
  protocol. It contains the implementation of the logic of the iterator exposed
  through the open/next/close iterator interface.

  At this level, the identity of the originating iterator op does not matter
  anymore and is lost. Iterator bodies may, in fact, not correspond to any
  particular iterator op, for example, after one iterator body has been inlined
  into another one.
