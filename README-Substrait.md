# Substrait Dialect for MLIR

This project consist of building an input/output dialect in
[MLIR](https://mlir.llvm.org/) for [Substrait](https://substrait.io/), the
cross-language serialization format of database query plans (akin to an
intermediate representation/IR for database queries). The immediate goal is to
create common infrastructure that can be used to implement consumers, producers,
optimizers, and transpilers of Substrait; the more transcending goal is to study
the viability of using modern, general-purpose compiler infrastructure to
implement database query compilers.

## Motivation

Substrait defines a serialization format for data-intensive compute operations
similar to relational algebra as they typically occur in database query plans
and similar systems, i.e., an exchange format for database queries. This allows
to separate the development of user frontends such as dataframe libraries or SQL
dialects (aka "Substrait producers") from that of backends such as database
engines (aka "Substrait consumers") and, thus, to interoperate more easily
between different data processing systems.

While Substrait has significant momentum and finds increasing
[adoption](https://substrait.io/community/powered_by/) in mature systems, it is
only concerned with implementing the *serialization format* of query plans, and
leaves the *handling* of that format and, hence, the *in-memory format* or
*intermediate representation* (IR) of plans up to the systems that adopt it.
This will likely lead to repeated implementation effort for everything else
required to deal with that intermediate representation, including
serialization/desiralization to and from text and other formats, a host-language
representation of the IR such as native classes, error and location tracking,
rewrite engines, rewrite rules, and pass management, common optimizations such
as common sub-expression elimination, and similar.

This project aims to create a base for any system dealing with Substrait by
building a "dialect" for Substrait in [MLIR](https://mlir.llvm.org/). In a way,
it aims to build an *in-memory* format for the concepts defined by Substrait,
for which the latter only describe their *serialization format*. MLIR is a
generic compiler framework providing infrastructure for writing compilers from
any domain, is part of the LLVM ecosystem, and has an [active
community](https://discourse.llvm.org/c/mlir/31) with
[adoption](https://mlir.llvm.org/users/) from researchers and industry across
many domains. It makes it easy to add new IR consisting of domain-specific
operations, types, attributes, etc., which are organized in dialects (either
in-tree and out-of-tree), as well as rewrites, passes, conversions,
translations, etc. on those dialects. Creating a Substrait dialect and a number
of common related transformations in such a mature framework has the potential
to eliminate some of the repeated effort described above and, thus, to ease and
eventually increase adoption of Substrait. By extension, building out a dialect
for Substrait can show that MLIR is a viable base for any database-style query
compiler.

## Target Use Cases

The aim of the Substrait dialect is to support all of the following use cases:

* Implement the **translation** of the IR of a particular system to or from
  Substrait by converting it to or from the Substrait dialect (rather than
  Substrait's protobuf messages) and then use the serialization/deserializing
  routines from this project.
* Use the Substrait dialect as the **sole in-memory format** for the IR of a
  particular system, e.g., parsing some frontend format into its own dialect
  and then converting that into the Substrait dialect for export or converting
  from the Substrait dialect for import and then translating that into an
  execution plan.
* Implement **simplifying and "canonicalizing" transformations** of Substrait
  plans such as common sub-expression elimination, dead code elimination,
  sub-query/common table-expression inlining, selection and projection
  push-down, etc., for example, as part of a producer, consumer, or transpiler.
* Implement **"compatibility rewrites"** that transforms plans that using
  features that are unsupported by a particular consumer into equivalent plans
  using features that it does support, for example, as part of a producer,
  consumer, or transpiler.
* [Stretch] Implement a full-blow *query optimizer* using the dialect for both
  logical and physical plans. It is not clear whether this should be done with
  this dialect or rather one or two additional ones that are specifically
  designed with query optimization in mind.

## Design Rationale

The main objective of the Substrait dialect is to allow handling Substrait plans
in MLIR: it replicates the components of Substrait plans as a dialect in order
to be able to tap into MLIR infrastructure. In the [taxonomy of Niu and
Amini](https://www.youtube.com/watch?v=hIt6J1_E21c&t=795s), this means that the
Substrait dialect is both an "input" and an "output" dialect for Substrait. As
such, there is only little freedom in designing the dialect. To guide the design
of the few remaining choices, we shall follow the following rationale (from most
important to least important):

* Every valid Substrait plan MUST be representable in the dialect.
* Every valid Substrait plan MUST round-trip through the dialect to the same
  plan as the input. This includes names and ordering.
* The import routine MUST be able to report all constraint violations of
  Substrait plans (such as type mismatches, dangling references, etc.).
* The dialect MAY be able to represent programs that do not correspond to valid
  Substrait plans. It MAY be impossible to export those to Substrait. For
  example, this allows to represent DAGs of operators rather than just trees.
* Every valid program in the Substrait dialect that can be exported to Substrait
  MUST round-trip through Substrait to a *semantically* equivalent program but
  MAY be different in terms of names, ordering, used operations, attributes,
  etc.
* The dialect SHOULD be understood easily by anyone familiar with Substrait. In
  particular, the dialect SHOULD use the same terminilogy as the Substrait
  specification wherever applicable.
* The dialect SHOULD follow MLIR conventions, idioms, and best practices.
* The dialect SHOULD reuse types, attributes, operations, and interfaces of
  upstream dialects wherever applicable.
* The dialect SHOULD allow simple optimizations and rewrites of Substrait
  plans without requiring other dialects.
* The serialization of the dialect (aka its "assembly") MAY change over time.
  (In other words, the dialect is not meant as an exchange format between
  systems -- that's what Substrait is for.)

## Features (Inherited by MLIR)

MLIR provides infrastructure for virtually all aspects of writing a compiler.
The following is a list of features that we inherit by using MLIR:

* Mostly declarative approach to defining relations and expressions (via
  [ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/)/tablegen).
* Documentation generation from declared relations and expressions (via
  [ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-documentation)).
* Declarative serialization/parsing to/from human-readable text representation
  (via [custom
  assembly](https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format)).
* Syntax high-lighting, auto-complete, as-you-type diagnostics, code navigation,
  etc. for the MLIR text format (via an [LSP
  server](https://mlir.llvm.org/docs/Tools/MLIRLSP/)).
* (Partially declarative) type deduction framework (via [ODS
  constraints](https://mlir.llvm.org/docs/DefiningDialects/Operations/#constraints)
  or C++
  [interface](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/InferTypeOpInterface.td)
  implementations).
* (Partially declarative) verification of arbitrary consistency constraints,
  declarative (via [ODS
  constraints](https://mlir.llvm.org/docs/DefiningDialects/Operations/#constraints))
  or imperative (via [C++
  verifiers](https://mlir.llvm.org/docs/DefiningDialects/Operations/#custom-verifier-code)).
* Mostly declarative pass management (via
  [tablegen](https://mlir.llvm.org/docs/PassManagement/#declarative-pass-specification)).
* Versatile infrastructure for pattern-based rewriting (via
  [DRR](https://mlir.llvm.org/docs/DeclarativeRewrites/) and [C++
  classes](https://mlir.llvm.org/docs/PatternRewriter/)).
* Powerful manipulation of imperative handling, creation, and modification of IR
  using [native
  classes](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/#op-vs-operation-using-mlir-operations)
  for operations, types, and attributes,
  [walkers](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/#walkers),
  [builders](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Builders.h),
  (IR) [interfaces](https://mlir.llvm.org/docs/Interfaces/), etc. (via ODS and
  C++ infrastructure).
* Powerful
  [location](https://mlir.llvm.org/docs/Dialects/Builtin/#location-attributes)
  tracking and location-based error reporting.
* Generated [Python bindings](https://mlir.llvm.org/docs/Bindings/Python/) of IR
  components, passes, and generic infrastructure (via ODS).
* Powerful command line argument handling and customizable implementation of
  typical [tools](https://github.com/llvm/llvm-project/tree/main/mlir/tools)
  (`X-opt`, `X-translate`, `X-lsp-server`, ...).
* [Testing infrastructure](https://mlir.llvm.org/getting_started/TestingGuide/)
  that is optimized for compilers (via `lit` and `FileCheck`).
* A collection of [common types and
  attributes](https://mlir.llvm.org/docs/Dialects/Builtin/) as well as
  [dialects](https://mlir.llvm.org/docs/Dialects/) (i.e., operations) for more
  or less generic purposes that can be used in or combined with custom dialects
  and that come with [transformations](https://mlir.llvm.org/docs/Passes/) on
  and [conversions](https://mlir.llvm.org/docs/DialectConversion/) to/from other
  dialects.
* A collection of
  [interfaces](https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir/Interfaces)
  and transformation passes on those interfaces, which allows to extend existing
  transformations to new dialects easily.
* A support library with efficient data structures, platform-independent file
  system abstraction, string utilities, etc. (via
  [MLIR](https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir/Support)
  and
  [LLVM](https://github.com/llvm/llvm-project/tree/main/llvm/include/llvm/Support)
  support libraries).
