# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Schedules for JAX.

To compile a function using Jasc, use Jasc.jit instead of jax.jit.

```
def computation(a: jax.Array) -> jax.Array:
  return jasc.tag(lambda x: x + 1, "plus1")(a)

def schedule(h: OpHandle) -> None:
  h.match_tag("plus1").tile((4,))

jasc.jit(computation, schedule)
```
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Iterable, Sequence
import contextlib
from dataclasses import dataclass, field, fields
from enum import Enum
import itertools
from typing import (
    Any,
    ClassVar,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    overload,
)

import jax
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import pdl, transform
from jaxlib.mlir.dialects.bufferization import LayoutMapOption
from jaxlib.mlir.dialects.transform import (
    bufferization,
    gpu,
    loop,
    memref,
    nvgpu,
    sparse_tensor,
    structured,
)

import call_kernel
from dialect import jasc as jasc_dialect
import primitives
from transform_ops import jasc_transform_ops as jto

_JASC_AUTO_NORMALIZATION = True


def set_auto_normalization(activate: bool):
  """Toggles the automatic normalization mode."""
  global _JASC_AUTO_NORMALIZATION
  _JASC_AUTO_NORMALIZATION = activate


@contextlib.contextmanager
def autonormalize(activate: bool = True):
  """Context manager that switches automatic normalization behavior."""
  autonorm_enabled: bool = _JASC_AUTO_NORMALIZATION
  set_auto_normalization(activate)
  try:
    yield
  finally:
    set_auto_normalization(autonorm_enabled)


StaticIntLike = Union[int, ir.IntegerAttr]
StaticBoolLike = Union[bool, ir.BoolAttr]
ValueLike = Union[ir.Operation, ir.OpView, ir.Value]
MixedInt = Union[StaticIntLike, ValueLike]

IntOrAttrList = Sequence[StaticIntLike]
OptionalIntList = Optional[Union[ir.ArrayAttr, IntOrAttrList]]

BoolOrAttrList = Sequence[StaticBoolLike]
OptionalBoolList = Optional[Union[ir.ArrayAttr, BoolOrAttrList]]

DynamicIndexList = Sequence[Union[MixedInt, Sequence[MixedInt]]]


@dataclass
class MultiHandleResult(abc.ABC):
  """Base class for all classes that support returning named handles."""

  def __iter__(self):
    yield from [getattr(self, field.name) for field in fields(self)]


@dataclass
class AllocaToGlobalResult(MultiHandleResult):
  get_global: OpHandle
  global_: OpHandle


@dataclass
class BufferizatToAllocationResult(MultiHandleResult):
  allocated_buffer: ValueHandle
  new_ops: OpHandle


@dataclass
class ForeachResult(MultiHandleResult):
  _op: Any

  @property
  @contextlib.contextmanager
  def body(self) -> OpHandle:
    """Creates a context manager with an insertion point on the body block.

    Yields an `OpHandle` of the argument of the body block. Inserts a `YieldOp`
    at the end of the body block if none is present.
    """
    # Set insertion point to body block and yield its argument.
    block = self._op.body.blocks[0]
    with ir.InsertionPoint(block):
      yield OpHandle(block.arguments[0])

    # Add `yield` to body if none present.
    operations = block.operations
    if len(operations) == 0 or not isinstance(
        operations[len(operations) - 1], transform.YieldOp
    ):
      with ir.InsertionPoint(block):
        transform.YieldOp()

  @property
  def results(self) -> list[OpHandle]:
    return [OpHandle(result) for result in self._op.results_]


@dataclass
class MapCopyToThreadsResult(MultiHandleResult):
  forall_op: OpHandle
  tiled_op: OpHandle


@dataclass
class PadResult(MultiHandleResult):
  padded: OpHandle
  pad: OpHandle
  copy: OpHandle


class PadCopyBackOp(Enum):
  NONE = "none"
  LINALG_COPY = "linalg.copy"
  BUFFER_COPY_TENSOR = "bufferization.copy_tensor"


class TileLoopKind(Enum):
  """Kind of loop operation to produce in tiling."""

  FOR = "scf.for"
  FORALL = "scf.forall"


@dataclass
class TileResult(MultiHandleResult):
  tiled_op: OpHandle
  loops: Sequence[OpHandle]


@dataclass(frozen=True)
class Normalform(abc.ABC):
  """Base class for all normalforms.

  A normalform is defined through a sequence of transformations to be applied to
  a handle to reach this normalform.
  """

  propagate_up: ClassVar[bool]
  propagate_down: ClassVar[bool]

  @classmethod
  @abc.abstractmethod
  def _impl(cls, handle: Value) -> Value:
    """Defines the transformations required to reach this normalform.

    A normalform may apply arbitrary transforms as long as `handle` is updated
    to wrap a valid mlir transform handle. This means a normalform might consume
    the initial MLIR transform handle and update `handle` to represent a
    different type of operation. Child handles of `handle` should be updated if
    that makes sense semantically, but may be invalidated in the process of
    normalization.
    """
    ...
    # TODO(@mluecke): Add tracking of handle invalidations so we can report if
    #                 an invalid handle is accessed after e.g. normalization.

  @classmethod
  def apply(cls, handle: Value) -> Value:
    """Apply transformations to a handle to bring it into this normalform."""
    new_handle = cls._impl(handle)
    # Setting this property propagates the normalform accordingly
    handle.normalform = cls
    return new_handle


@dataclass(frozen=True)
class AnyForm(Normalform):
  """Weakest normal form. Any program can be considered to be in this form."""

  propagate_up: ClassVar[bool] = True
  propagate_down: ClassVar[bool] = True

  @classmethod
  def _impl(cls, handle: Value) -> Value:
    return handle


@dataclass(frozen=True)
class LoopNormalform(Normalform):
  """A normal form that enables most loop based transformations.

  This normalform can only be applied to an OpHandle.
  """

  propagate_up: ClassVar[bool] = False
  propagate_down: ClassVar[bool] = True

  @classmethod
  def _impl(cls, handle: OpHandle) -> OpHandle:
    with handle.apply_patterns():
      structured.ApplyTilingCanonicalizationPatternsOp()
      jto.ApplyFoldFillIntoPadPatternsOp()
      loop.ApplyForLoopCanonicalizationPatternsOp()
      transform.ApplyCanonicalizationPatternsOp()

    handle.apply_licm(["scf.for"])
    handle.apply_cse()
    return handle


C = TypeVar("C", bound=Callable)


def jasc_transform(
    enforced_normalform: Optional[Type[Normalform] | C] = AnyForm,
    required_normalform: Optional[Type[Normalform]] = AnyForm,
    no_propagate: Optional[bool] = False,
) -> C:
  """Decorator for jasc abstractions adding automatic handling of normalization.

  Args:
    enforced_normalform: The normalform the resulting handles will have.
    required_normalform: The required normalform to apply this transform.
    no_propagate: If true, no changes to any normalforms will be done.

  Returns:
    The decorated function according to the following:

  This enables automatic enforcement of a specific normalform before this
  transform is executed. Propagates the enforced/retained normalform to the
  resulting handles.
  If no explicit normalform is provided the handles are conservatively assumed
  to now be in AnyForm, i.e. the weakest normalform.
  """

  def wrapped(f: C) -> C:
    def decorated(*args, **kwargs):
      if required_normalform:
        # TODO(@mluecke): this assumes that the payload op is surrounded by a
        # func op that will be matched and normalized. This is not always
        # guaranteed to be the case.
        args[0].auto_normalize_parent_func(required_normalform)
      results = f(*args, **kwargs)

      def flatten(results: Any) -> List:
        """Unpacks all potentially nested iterables into a flat list."""
        all_results = [results]
        is_iterable = lambda x: isinstance(x, Iterable)
        while any(is_iterable(x) for x in all_results):
          all_results = list(
              itertools.chain.from_iterable(
                  x if is_iterable(x) else [x] for x in all_results
              )
          )
        return all_results

      if not no_propagate:
        for result in flatten(results):
          result.normalform = enforced_normalform
      return results

    return decorated

  # If the decorator was used without `()` the decorated function will be in
  # this variable. We remap it and reset it to the default value. This enables
  # using this decorator in similar fashion to e.g. the `dataclass` decorator.
  if not isinstance(enforced_normalform, Type):
    f = enforced_normalform
    enforced_normalform = AnyForm
    return wrapped(f)
  return wrapped


def tuning_param(default_value: Optional[ir.Attribute | int] = None) -> Param:
  """Emits a transform op that provides an "empty" to be autotuned param.

  Args:
    default_value: The default value for this parameter. This will be used
      during interpretation of the transform IR if no autotuning is performed.
      If not specfied, 1 : i32 will be used as default value.
  """
  i32_type = ir.IntegerType.get_signless(32)
  if default_value is None:
    default_value = ir.IntegerAttr.get(i32_type, 1)
    param_type = transform.ParamType.get(i32_type)
  elif isinstance(default_value, int):
    default_value = ir.IntegerAttr.get(i32_type, default_value)
    param_type = transform.ParamType.get(i32_type)
  elif isinstance(default_value.type, ir.IntegerType):
    # Support explicit param type for int types of different widths / signedness
    param_type = transform.ParamType.get(default_value.type)
  else:
    # TODO(mluecke): Make this more general once transform.ParamType supports
    #                types beyond IntegerType
    param_type = transform.AnyParamType.get()
  op = jto.TuningParamOp(param_type, default_value)
  return Param(op.param)


def constant_param(value: ir.Attribute | int) -> Param:
  """Emits a transform op that provides a constant param."""
  if isinstance(value, int):
    value = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)
    param_type = transform.ParamType.get(value.type)
  else:
    # TODO(mluecke): Make this more general once transform.ParamType supports
    #                types beyond IntegerType
    param_type = transform.AnyParamType.get()
  op = transform.ParamConstantOp(param_type, value)
  return Param(op.param)


@dataclass
class Value(abc.ABC):
  """Wrapper around a transform handle with methods to chain further transforms."""

  _mlir_value: ir.Value
  _normalform: Type[Normalform] = AnyForm
  children: list[Value] = field(default_factory=list)
  parent: Optional[Value] = None

  @property
  def mlir_value(self) -> ir.Value:
    return self._mlir_value

  @property
  def normalform(self) -> Type[Normalform]:
    return self._normalform

  @normalform.setter
  def normalform(self, normalform: Type[Normalform]):
    self._normalform = normalform
    if self._normalform.propagate_up:
      self.propagate_up_normalform(normalform)
    if self._normalform.propagate_down:
      self.propagate_down_normalform(normalform)

  def propagate_up_normalform(self, normalform: Type[Normalform]):
    if self.parent:
      # Using the property here would trigger infinite propagation for NFs that
      # have to be propagated up and down
      self.parent._normalform = normalform
      self.parent.propagate_up_normalform(normalform)

  def propagate_down_normalform(self, normalform: Type[Normalform]):
    for child in self.children:
      # Using the property here would trigger infinite propagation for NFs that
      # have to be propagated up and down
      child._normalform = normalform
      child.propagate_down_normalform(normalform)

  def normalize(self, normalform: Type[Normalform]) -> Value:
    """Applies transformations to bring this handle into a specific normalform."""
    normalform.apply(self)
    return self

  @classmethod
  def _unwrap_handles_from_dynamic_index_list(
      cls,
      indices: Union[DynamicIndexList, ir.ArrayAttr],
  ) -> Union[DynamicIndexList, ir.ArrayAttr]:
    """Extracts the MLIR value from each OpHandle in the given DynamicIndexList.

    This brings it into the definition of `DynamicIndexList` used by the
    upstream op constructors, such that it can be passed as an argument there.
    """
    if indices is None:
      return None
    # ArrayAttr: there are no OpHandles inside, so nothing to do.
    if isinstance(indices, ir.ArrayAttr):
      return indices

    # It must be a list: process each index at a time.
    def extract_handle(index: Any) -> Any:
      if isinstance(index, Value):
        return index.mlir_value
      elif not isinstance(index, (StaticIntLike, ValueLike)):
        # If it's not one of these types, it must be a scalable index, which is
        # a singleton list of one index.
        return [extract_handle(index[0])]
      return index

    return [extract_handle(index) for index in indices]


@dataclass
class Param(Value):
  """Wrapper around a transform Param with methods to chain further transforms."""


@dataclass
class OpHandle(Value):
  """Wrapper around an OpHandle with methods to chain further transforms."""

  def _ensure_op_type(self, type: Union[ir.Type, str]) -> OpHandle:
    """Returns a handle to the same payload ops with the given op type.

    If the op type of the given handle already corresponds to the given type,
    it is returned as is. Otherwise, a `transform.cast` is inserted and a handle
    to that op is returned. If the expected type is given as a `str`, it is used
    to construct a `transform.op<...>` with that string for the expected type.
    """
    if isinstance(type, str):
      type = transform.OperationType.get(type)

    if self.mlir_value.type != type:
      return self.cast(type)
    else:
      return self

  def alloca_to_global(self) -> AllocaToGlobalResult:
    """Creates a `MemRefAllocaToGlobalOp` and returns handles with the results.

    This handle will be updated to represent the tiled newly inserted
    `memref.global` ops.
    """
    alloca = self._ensure_op_type("memref.alloca")
    op = memref.MemRefAllocaToGlobalOp(alloca.mlir_value)
    self._mlir_value = op.getGlobal
    return AllocaToGlobalResult(get_global=self, global_=OpHandle(op.global_))

  def apply_cse(self) -> OpHandle:
    """Creates a `ApplyCommonSubexpressionEliminationOp` and returns `self`."""
    transform.ApplyCommonSubexpressionEliminationOp(self.mlir_value)
    return self

  def apply_dce(self) -> OpHandle:
    """Creates a `ApplyDeadCodeEliminationOp` and returns `self`."""
    transform.ApplyDeadCodeEliminationOp(self.mlir_value)
    return self

  def apply_licm(
      self, to: Optional[Sequence[str | OpHandle]] = None
  ) -> OpHandle:
    """Creates a `ApplyLoopInvariantCodeMotionOp` for each given op and returns `self`.

    For strings in `to`, matches ops with these names. For the resulting matches
    and for each `Value` given directly in `to`, a LICM transform op
    (`transform.ApplyLoopInvariantCodeMotionOp`) is created. If `to` is `None`,
    a LICM is created for `self`.
    """
    # Handle `self` case.
    if to is None:
      to = [self]

    # Create match for string inputs.
    op_names = [op for op in to if isinstance(op, str)]
    ops = [op for op in to if isinstance(op, OpHandle)]
    if op_names:
      matched = self.match_ops(op_names)
      ops.append(matched)

    # Create LICM ops.
    for op in ops:
      transform.ApplyLoopInvariantCodeMotionOp(op.mlir_value)

    return self

  @contextlib.contextmanager
  def apply_patterns(self, *, apply_cse: Optional[bool] = None):
    """Emits a `transform.ApplyPatternsOp`.

    Returns a context manager with an insertion point on the patterns block.
    """
    op = transform.ApplyPatternsOp(self.mlir_value)
    op.apply_cse = apply_cse
    with ir.InsertionPoint(op.patterns):
      yield

  def apply_tuning_config(
      self, config: Sequence[int | ir.Attribute]
  ) -> OpHandle:
    """Creates a `ApplyTuningConfigOp` and returns `self`."""
    config_attr = ir.ArrayAttr.get(
        [
            ir.IntegerAttr.get(ir.IntegerType.get_signless(32), i)
            if isinstance(i, int)
            else i
            for i in config
        ]
    )
    jto.ApplyTuningConfigOp(self.mlir_value, config=config_attr)
    return self

  def auto_normalize_parent_func(self, normalform: Type[Normalform]):
    """Auto normalizes the parent function if needed."""
    if self.normalform != normalform and _JASC_AUTO_NORMALIZATION:
      func = self.get_parent_op(op_name="func.func", deduplicate=True)
      func.normalize(normalform)

  def buffer_loop_hoisting(self) -> OpHandle:
    """Creates a `bufferization.BufferLoopHoistingOp` and returns `self`."""
    bufferization.BufferLoopHoistingOp(self.mlir_value)
    return self

  def bufferize_to_allocation(
      self,
      *,
      memory_space: Optional[int | str | ir.Attribute] = None,
      memcpy_op: Optional[str] = None,
      alloc_op: Optional[str] = None,
      bufferize_destination_only: Optional[bool] = None,
  ) -> BufferizatToAllocationResult:
    """Creates a `structured.BufferizeToAllocationOp` op.

    Returns the results as handles in a `BufferizeToAllocationResult`.
    """
    op = structured.BufferizeToAllocationOp(
        self.mlir_value,
        memory_space=memory_space,
        memcpy_op=memcpy_op,
        alloc_op=alloc_op,
        bufferize_destination_only=bufferize_destination_only,
    )

    return BufferizatToAllocationResult(
        allocated_buffer=ValueHandle(op.allocated_buffer),
        new_ops=OpHandle(op.new_ops),
    )

  def cast(self, type_: ir.Type | str) -> OpHandle:
    """Creates a handle from the result of a `CastOp` to the given type."""
    if isinstance(type_, str):
      type_ = transform.OperationType.get(type_)
    op = transform.CastOp(type_, self.mlir_value)
    return OpHandle(op.output)

  def create_async_groups(self) -> OpHandle:
    """Creates a handle from the result of a new `CreateAsyncGroupsOp` op."""
    op = nvgpu.CreateAsyncGroupsOp(transform.AnyOpType.get(), self.mlir_value)
    # XXX: Should self._mlir_value be updated?
    return OpHandle(op.result)

  def eliminate_empty_tensors(self) -> OpHandle:
    """Creates a `bufferization.EliminateEmptyTensorsOp` and returns `self`."""
    bufferization.EliminateEmptyTensorsOp(self.mlir_value)
    return self

  def foreach(
      self, result_types: Optional[Union[ir.Type, Sequence[ir.Type]]] = None
  ):
    """Emits a `transform.foreach` op.

    The result object gives access to a context manager with an insertion point
    on the body block as well as the results of the op.
    """
    # TODO(ingomueller): Move boilerplate to upstream and make upstream `body`
    #                    property return first block.
    if result_types is None:
      result_types = []
    if isinstance(result_types, ir.Type):
      result_types = [result_types]

    input_type = self.mlir_value.type
    op = transform.ForeachOp(results_=result_types, target=self.mlir_value)
    op.body.blocks.append(input_type)

    return ForeachResult(_op=op)

  @jasc_transform(required_normalform=LoopNormalform)
  def fuse_into(
      self, containing_op: Union[ir.Operation, ir.OpView, ir.Value]
  ) -> OpHandle:
    """Creates a new `structured.FuseIntoContainingOp`.

    The func.func payload op surrounding the payload this handle represents
    will be autonormalized to LoopNormalform if needed.

    This handle will afterwards point to the `fused_op` result. The
    `containing_op` handle remains valid.
    """
    op = structured.FuseIntoContainingOp(
        self.mlir_value, containing_op.mlir_value
    )
    self._mlir_value = op.fused_op
    return self

  def get_parent_op(
      self,
      deduplicate: Optional[StaticBoolLike] = None,
      isolated_from_above: Optional[StaticBoolLike] = None,
      op_name: Optional[str] = None,
  ) -> OpHandle:
    """Creates a handle from the result of a new `GetParentOp` op."""
    op = transform.GetParentOp(
        transform.AnyOpType.get(),
        self.mlir_value,
        deduplicate=deduplicate,
        isolated_from_above=isolated_from_above,
        op_name=op_name,
    )
    return OpHandle(op.parent)

  def get_producer_of_operand(
      self, operand_number: int | ir.Attribute
  ) -> OpHandle:
    """Creates a handle from the result of a new `GetProducerOfOperand` op."""
    op = transform.GetProducerOfOperand(
        transform.AnyOpType.get(), self.mlir_value, operand_number
    )
    return OpHandle(op.producer)

  def hoist_pad(self, num_loops: int | ir.Attribute) -> OpHandle:
    """Creates a new `structured.HoistPadOp` op.

    This handle will be updated to represent the result of the transform.
    """
    op = structured.HoistPadOp(
        transform.AnyOpType.get(), self.mlir_value, num_loops
    )
    self._mlir_value = op.transformed
    return self

  def hoist_redundant_vector_transfers(self) -> OpHandle:
    """Creates a new `structured.hoist_redundant_vector_transfers` op.

    This handle will be updated to represent the result of the transform.
    """
    op = structured.HoistRedundantVectorTransfersOp(
        transform.AnyOpType.get(), self.mlir_value
    )
    self._mlir_value = op.transformed
    return self

  def insert_slice_to_copy(self) -> OpHandle:
    """Creates a new `structured.InsertSliceToCopyOp` op.

    Updates this handle to represent the new linalg.copy operation.

    The transform is a targeted rewrite of a `tensor.insert_slice` or
    `tensor.parallel_insert_slice` to `linalg.copy`. If the insert_slice source
    is already a linalg.copy, only returns the source op (i.e. does not create
    an additional linalg.copy op).
    """
    op = structured.InsertSliceToCopyOp(
        transform.OperationType.get("linalg.copy"), self.mlir_value
    )
    self._mlir_value = op.transformed
    return self

  def interchange(
      self, iterator_interchange: OptionalIntList = None
  ) -> OpHandle:
    """Creates a new `structured.interchange` op.

    Updates this handle to represent the transformed linalg operation.
    """
    op = structured.InterchangeOp(
        self.mlir_value, iterator_interchange=iterator_interchange
    )
    return OpHandle(op.transformed)

  def map_forall_to_blocks(
      self,
      *,
      grid_dims: Optional[Union[Sequence[int], ir.Attribute]] = None,
      generate_gpu_launch: Optional[Union[bool, ir.Attribute]] = None,
  ) -> OpHandle:
    """Creates a new `gpu.MapForallToBlocks` op."""
    op = gpu.MapForallToBlocks(
        self.mlir_value,
        grid_dims=grid_dims,
        generate_gpu_launch=generate_gpu_launch,
    )
    self._mlir_value = op.result
    return self

  def map_copy_to_threads(
      self, *, total_num_threads: int, desired_bit_alignment: int
  ) -> MapCopyToThreadsResult:
    """Creates a new `structured.gpu.MapCopyToThreadsOp` op.

    This handle will be updated to represent the new tiled op.
    """
    op = structured.MapCopyToThreadsOp(
        self.mlir_value,
        total_num_threads=total_num_threads,
        desired_bit_alignment=desired_bit_alignment,
    )
    self._mlir_value = op.tiled_op
    return MapCopyToThreadsResult(
        forall_op=OpHandle(op.forall_op),
        tiled_op=self,
    )

  def map_nested_forall_to_threads(
      self,
      *,
      block_dims: OptionalIntList = None,
      sync_after_distribute: Optional[StaticBoolLike] = None,
      warp_size: OptionalIntList = None,
  ) -> OpHandle:
    """Creates a new `gpu.MapNestedForallToThreads` op."""
    op = gpu.MapNestedForallToThreads(
        transform.AnyOpType.get(),
        self.mlir_value,
        block_dims=block_dims,
        sync_after_distribute=sync_after_distribute,
        warp_size=warp_size,
    )
    self._mlir_value = op.result
    return self

  @jasc_transform(required_normalform=LoopNormalform)
  def vectorize(
      self,
      vector_sizes: Optional[Sequence[int | ir.Attribute]] = None,
      *,
      vectorize_nd_extract: Optional[bool] = None,
  ) -> OpHandle:
    """Creates a `structured.VectorizeOp` op and returns `self`.

    The func.func payload op surrounding the payload this handle represents
    will be autonormalized to LoopNormalform if needed.
    """
    if vector_sizes is not None:
      vector_sizes = self._unwrap_handles_from_dynamic_index_list(vector_sizes)
    structured.VectorizeOp(
        self.mlir_value,
        vector_sizes=vector_sizes,
        vectorize_nd_extract=vectorize_nd_extract,
    )
    return self

  def match_ops(
      self,
      ops: str
      | ir.OpView
      | structured.MatchInterfaceEnum
      | Sequence[str | ir.OpView],
  ) -> OpHandle:
    """Returns a handle to ops that match the given names, types, or interface.

    If only a single type is given, the value wrapped by the resulting
    handle is populated with the respective type.
    """
    # Handle interface.
    if isinstance(ops, structured.MatchInterfaceEnum) or (
        isinstance(ops, str)
        and ops in structured.MatchInterfaceEnum.__members__
    ):
      if isinstance(ops, str):
        ops = structured.MatchInterfaceEnum[ops]
      match_op = structured.MatchOp(
          transform.AnyOpType.get(),
          self.mlir_value,
          interface=ops,
      )

    # Handle op name(s), either given directly as string or given as op.
    else:
      if isinstance(ops, str):
        op_type = transform.OperationType.get(ops)
        op_names = [ops]
      elif isinstance(ops, Sequence):
        op_type = transform.AnyOpType.get()
        op_names = [
            op if isinstance(op, str) else op.OPERATION_NAME for op in ops
        ]
      else:
        op_type = transform.OperationType.get(ops.OPERATION_NAME)
        op_names = [ops.OPERATION_NAME]
      match_op = structured.MatchOp.match_op_names(
          op_type,
          self.mlir_value,
          op_names,
      )

    handle = OpHandle(match_op.results_, parent=self)
    self.children.append(handle)
    return handle

  def match_sparse_inout_ops(self) -> OpHandle:
    op_type = transform.AnyOpType.get()
    sparse_op = sparse_tensor.MatchSparseInOut(op_type, self.mlir_value)
    handle = OpHandle(sparse_op.result, parent=self)
    self.children.append(handle)
    return handle

  def match_tag(self, tag_names: str | Sequence[str]) -> OpHandle:
    """Returns a handle to linalg operations that match the given tags."""
    if isinstance(tag_names, str):
      tag_names = [tag_names]
    linalg_iface = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)
    linalg_ops = structured.MatchOp(
        pdl.OperationType.get(), self.mlir_value, interface=linalg_iface
    ).results
    match_tag_op = jto.MatchTagOp(linalg_ops, tags=tag_names)  # pylint: disable=no-value-for-parameter
    handle = OpHandle(match_tag_op.matched_ops, parent=self)
    self.children.append(handle)
    return handle

  def one_shot_bufferize(
      self,
      allow_return_allocs_from_loops: Optional[bool] = None,
      allow_unknown_ops: Optional[bool] = None,
      bufferize_function_boundaries: Optional[bool] = None,
      function_boundary_type_conversion: Optional[
          Enum | str | ir.Attribute
      ] = None,
      memcpy_op: Optional[str] = None,
      print_conflicts: Optional[bool] = None,
      test_analysis_only: Optional[bool] = None,
  ) -> OpHandle:
    """Creates a new `bufferization.OneShotBufferizeOp` op.

    This handle will be updated to represent the result of the transform.
    """
    if isinstance(function_boundary_type_conversion, str):
      function_boundary_type_conversion = LayoutMapOption[
          function_boundary_type_conversion
      ]

    op = bufferization.OneShotBufferizeOp(
        self.mlir_value,
        allow_return_allocs_from_loops=allow_return_allocs_from_loops,
        allow_unknown_ops=allow_unknown_ops,
        bufferize_function_boundaries=bufferize_function_boundaries,
        function_boundary_type_conversion=function_boundary_type_conversion,
        memcpy_op=memcpy_op,
        print_conflicts=print_conflicts,
        test_analysis_only=test_analysis_only,
    )
    self._mlir_value = op.transformed
    return self

  def pad(
      self,
      *,
      copy_back_op: Optional[Union[str, ir.StringAttr, PadCopyBackOp]] = None,
      pack_paddings: OptionalIntList = None,
      padding_dimensions: OptionalIntList = None,
      padding_values: Optional[Sequence[float]] = None,
      pad_to_multiple_of: OptionalIntList = None,
      transpose_paddings: Optional[
          Union[ir.ArrayAttr, Sequence[Union[ir.ArrayAttr, IntOrAttrList]]]
      ] = None,
  ) -> PadResult:
    """Creates a new `structured.PadOp` op.

    This handle will be updated to represent the new padded op.
    """

    if isinstance(copy_back_op, PadCopyBackOp):
      copy_back_op = copy_back_op.value

    if padding_values is not None:
      padding_values = ir.ArrayAttr.get(
          [ir.FloatAttr.get_f32(val) for val in padding_values]
      )

    op = structured.PadOp(  # pylint: disable=no-value-for-parameter
        self.mlir_value,
        copy_back_op=copy_back_op,
        pack_paddings=pack_paddings,
        padding_values=padding_values,
        padding_dimensions=padding_dimensions,
        pad_to_multiple_of=pad_to_multiple_of,
        transpose_paddings=transpose_paddings,
    )
    self._mlir_value = op.padded
    return PadResult(
        padded=self,
        pad=OpHandle(op.pad),
        copy=OpHandle(op.copy),
    )

  def print(self, name: Optional[str] = None) -> OpHandle:
    """Emits a transform op to print this handle and an optional message."""
    transform.PrintOp(target=self.mlir_value, name=name)
    return self

  def rewrite_in_destination_passing_style(self) -> OpHandle:
    """Creates a new `structured.RewriteInDestinationPassingStyleOp` op.

    This handle will be updated to represent the result of the transform.
    """
    op = structured.RewriteInDestinationPassingStyleOp(
        transform.AnyOpType.get(), self.mlir_value
    )
    self._mlir_value = op.transformed
    return self

  def select(self, op_name: str | ir.Attribute) -> OpHandle:
    """Returns a handle to the result of a new `transform.SelectOp`."""
    op = transform.SelectOp(transform.AnyOpType.get(), self.mlir_value, op_name)
    return OpHandle(op.result)

  def synchronize(self) -> OpHandle:
    """Creates a new `SynchronizeOp` op and returns a handle to the barrier.

    self will only be read by this transform and hence stay valid.
    """
    op = jto.SynchronizeOp(
        transform.OperationType.get("gpu.barrier"), self.mlir_value
    )
    return OpHandle(op.barrier)

  @jasc_transform(required_normalform=LoopNormalform)
  def take_assumed_branch(self, take_else_branch: bool = None) -> OpHandle:
    """Creates a `TakeAssumedBranchOp` and returns `self`.

    The func.func payload op surrounding the payload this handle represents
    will be autonormalized to LoopNormalform if needed.
    """
    loop.TakeAssumedBranchOp(self.mlir_value, take_else_branch=take_else_branch)
    return self

  def _tile_using_for(
      self,
      *,
      tile_sizes: Sequence[int | Param],
      interchange: Optional[Sequence[int]] = None,
  ):
    op = structured.TileUsingForOp(
        self.mlir_value,
        sizes=self._unwrap_handles_from_dynamic_index_list(tile_sizes),
        interchange=interchange,
    )
    self._mlir_value = op.tiled_linalg_op
    return TileResult(
        tiled_op=self,
        loops=[OpHandle(loop) for loop in op.loops],
    )

  @jasc_transform(required_normalform=LoopNormalform)
  def tile(
      self,
      *,
      loop: TileLoopKind,
      tile_sizes: Optional[Sequence[int | Param]] = None,
      interchange: Optional[Sequence[int]] = None,
      num_threads: Optional[Sequence[int]] = None,
      mapping: Optional[
          str | ir.Attribute | Sequence[str | ir.Attribute]
      ] = None,
  ) -> TileResult:
    """Creates a new structured tiling operation.

    Depending on the `loop` kwarg, creates either a `structured.TileUsingFor` or
    `structured.TileUsingForall` transform operation. Additional kwargs
    parameterize the created op:

    `tile_sizes`: tile sizes to use in the loop, mandatory for `for` loops;
    `num_threads`: the number of iterations in the produced loop, only supported
                   in `forall` tiling at the moment;
    `interchange`: interchange of the dimensions, only supported in `for` tiling
                   at the moment;
    `mapping`: mapping of the generated loops to parallelism concepts such as
               GPU threads, only supported in `forall` loops (`for` loops are
               implicitly sequential).

    This handle will be updated to represent the tiled linalg op.
    """
    if loop == TileLoopKind.FOR:
      if tile_sizes is None:
        raise ValueError("Tile sizes must be provided.")
      if num_threads is not None or mapping is not None:
        raise ValueError(
            "Cannot specify num threads or mapping when tiling to scf.for, use"
            " scf.forall instead."
        )
      return self._tile_using_for(
          tile_sizes=tile_sizes, interchange=interchange
      )

    elif loop == TileLoopKind.FORALL:
      if tile_sizes is None and num_threads is None:
        raise ValueError("Must specify either tile sizes or num threads.")
      if interchange is not None:
        raise ValueError(
            "Cannot specify interchange when tiling to scf.forall."
        )
      if tile_sizes and any(
          isinstance(tile_size, Param) for tile_size in tile_sizes
      ):
        raise ValueError(
            "Cannot specify dynamic tile sizes when tiling to scf.forall."
        )
      return self._tile_using_forall(
          tile_sizes=tile_sizes, num_threads=num_threads, mapping=mapping
      )

    raise ValueError(f"Uknown loop kind {loop}")

  def replace_with_alloc_tensor(self) -> OpHandle:
    """Creates a new `bufferization.EmptyTensorToAllocTensorOp` and updates this handle accordingly.

    The payload op in this handle has to be a `tensor.empty`. If the static type
    of the MLIR handle indicates that this is the case, the handle is used as
    is; otherwise, a `transform.cast` op is inserted that casts this MLIR handle
    into the required type.
    """

    tensor_empty = self._ensure_op_type("tensor.empty").mlir_value
    op = bufferization.EmptyTensorToAllocTensorOp(tensor_empty)
    self._mlir_value = op.transformed
    return self

  def _tile_using_forall(
      self,
      *,
      mapping: Optional[
          str | ir.Attribute | Sequence[str | ir.Attribute]
      ] = None,
      num_threads: Optional[Sequence[int]] = None,
      tile_sizes: Optional[Sequence[int]] = None,
  ) -> TileResult:
    """Creates a new `structured.TileUsingForallOp` op.

    The func.func payload op surrounding the payload this handle represents
    will be autonormalized to LoopNormalform if needed.

    This handle will be updated to represent the tiled op.
    """
    # TODO(mluecke): Remove string parsing of attributes once builders for GPU
    #                dialect attributes are available
    attr_or_parse = lambda x: ir.Attribute.parse(x) if isinstance(x, str) else x
    if isinstance(mapping, (str, ir.Attribute)):
      mapping = attr_or_parse(mapping)
    elif mapping is not None:
      mapping = ir.ArrayAttr.get([attr_or_parse(attr) for attr in mapping])

    op = structured.TileUsingForallOp(
        transform.AnyOpType.get(),
        transform.AnyOpType.get(),
        self.mlir_value,
        num_threads=num_threads,
        tile_sizes=tile_sizes,
        mapping=mapping,
    )
    self._mlir_value = op.tiled_op
    return TileResult(
        loops=[OpHandle(op.forall_op)],
        tiled_op=self,
    )

  def vectorize_children_and_apply_patterns(
      self,
      disable_multi_reduction_to_contract_patterns: Optional[bool] = None,
      disable_transfer_permutation_map_lowering_patterns: Optional[bool] = None,
      vectorize_nd_extract: Optional[bool] = None,
      vectorize_padding: Optional[bool] = None,
  ) -> OpHandle:
    """Creates a new `structured.VectorizeChildrenAndApplyPatternsOp` op.

    This handle will be updated to represent the result of the transform.
    """
    op = structured.VectorizeChildrenAndApplyPatternsOp(  # pylint: disable=no-value-for-parameter
        self.mlir_value,
        disable_multi_reduction_to_contract_patterns=disable_multi_reduction_to_contract_patterns,
        disable_transfer_permutation_map_lowering_patterns=disable_transfer_permutation_map_lowering_patterns,
        vectorize_nd_extract=vectorize_nd_extract,
        vectorize_padding=vectorize_padding,
    )
    self._mlir_value = op.transformed
    return self


@dataclass
class ValueHandle(Value):
  """Wrapper around a ValueHandle with methods to chain further transforms."""


class Schedule(Protocol):
  """A schedule for a Jax computation.

  Example:
  ```
  def computation(a: jax.Array) -> jax.Array:
    return jasc.tag(lambda x: x + 1, "plus1")(a)

  def schedule(h: OpHandle) -> None:
    h.match_tag("plus1").tile((4,))

  jasc.jit(computation, schedule)
  ```
  """

  def __call__(self, handle: OpHandle) -> None:
    """Builds a schedule for the computation handle to point to."""
    ...


def _flatten_func(
    func: Callable[..., Any], args: Sequence[Any]
) -> tuple[Callable[..., Any], Sequence[jax.Array]]:
  """Flattens a function inputs and ouputs.

  See jax.tree_util.tree_flatten for background information on flattening.

  Args:
    func: A function to flatten.
    args: Arguments that will be passed to func.

  Returns:
    A tuple composed of:
    - a function that has the same semantics than func, but with inputs and
      outputs flattened.
    - args flattened.
  """
  flat_args, in_tree = jax.tree_util.tree_flatten(args)

  def flat(*flat_args: Any) -> Any:
    unflat_args = jax.tree_util.tree_unflatten(in_tree, flat_args)
    out = func(*unflat_args)
    flat_outs, _ = jax.tree_util.tree_flatten(out)
    return flat_outs

  return flat, flat_args


def jit(
    func: Callable[..., Any],
    schedule: Optional[Schedule] = None,
    *,
    module: Optional[ir.Module] = None,
    dump_ir: bool = False,
) -> Callable[..., Any]:
  """Applies scheduling directives inside func.

  Args:
    func: A function to compile using Jasc. The function must use Jasc schedule
      directives to optimize its code.
    schedule: A schedule to apply to the computation.
    module: An already lowered representation of `func` in MLIR. If this is
      supplied it will be used for execution rather than lowering `func`.
    dump_ir: If true, logs intermediate compilation steps.

  Returns:
    A function with the same semantics as func, but compiled using Jasc.
  """

  @jax.jit
  def wrapped(*args: Any) -> Any:
    if schedule is None:
      build_schedule = lambda handle: handle
    else:
      build_schedule = lambda handle: schedule(OpHandle(handle))
    flat_func, flat_args = _flatten_func(func, args)
    out_avals = jax.tree_map(
        lambda x: jax.core.ShapedArray(x.shape, x.dtype),
        jax.eval_shape(func, *args),
    )
    flat_out_avals, out_tree = jax.tree_util.tree_flatten(out_avals)
    out_flat = primitives.jit_p.bind(
        *flat_args,
        func=flat_func,
        module=module,
        build_schedule=build_schedule,
        out_avals=flat_out_avals,
        dump_ir=dump_ir,
    )
    return jax.tree_util.tree_unflatten(out_tree, out_flat)

  return wrapped


def apply_schedule(
    module: ir.Module,
    schedule: Schedule | None = None,
    dump_ir: bool = False,
    dump_schedule: bool = False,
) -> None:
  """Applies a schedule to the module.

  Args:
    module: Existing module with payload IR and possibly an existing schedule.
    schedule: The schedule to apply at linalg level. If no schedule is supplied
      it is assumed to already be present in the module.
    dump_ir: Whether to dump the transformed IR after each pass.
    dump_schedule: Whether to dump the schedule after creation. This is only
      supported with a schedule that is not already in the module.
  """

  if schedule is not None:
    insert_schedule(module, schedule, dump_schedule)
  if schedule is None and dump_schedule:
    raise ValueError(
        "dump_schedule is only supported with a schedule that is not already in"
        " the module."
    )
  call_kernel.apply_schedule(module, dump_ir)


def insert_schedule(
    module: ir.Module,
    schedule: Schedule | None = None,
    dump_schedule: bool = False,
) -> None:
  """Inserts the transform script of the schedule into the module.

  Args:
    module: Existing module into which the script should be inserted.
    schedule: The schedule to apply at linalg level.
    dump_schedule: Whether to dump the schedule after creation.
  """
  if schedule is None:
    schedule = lambda x: x

  # Register jasc transform ops so they can be used in the schedule
  jasc_dialect.register_and_load_dialect(module.context)

  # Insert the schedule into the IR
  with module.context, ir.Location.unknown(module.context):
    with ir.InsertionPoint.at_block_begin(module.body):
      sequence_op = transform.SequenceOp(
          transform.FailurePropagationMode.Propagate,
          (),
          transform.AnyOpType.get(),
      )
    with ir.InsertionPoint(sequence_op.body):
      schedule(OpHandle(sequence_op.bodyTarget))
      transform.YieldOp([])

  if dump_schedule:
    print(sequence_op)


def tag(func: Callable[..., Any], name: str) -> Callable[..., Any]:
  """Tags the function so that it can be matched by Jasc schedules.

  Args:
    func: The function to tag.
    name: Identifier of the tag. If the same name is used in multiple places,
      Jasc schedules will match all occurences of the name.

  Returns:
    A function with the same semantics as func.
  """

  def wrapped(*args: Any) -> Any:
    flat_func, flat_args = _flatten_func(func, args)
    out_avals = jax.tree_map(
        lambda x: jax.core.ShapedArray(x.shape, x.dtype),
        jax.eval_shape(func, *args),
    )
    flat_out_avals, out_tree = jax.tree_util.tree_flatten(out_avals)
    out_flat = primitives.tag_p.bind(
        *flat_args,
        func=flat_func,
        out_avals=flat_out_avals,
        name=name,
    )
    return jax.tree_util.tree_unflatten(out_tree, out_flat)

  return wrapped


def yield_(values: Optional[Union[Value, Sequence[Value]]] = None) -> None:
  if values is None:
    values = []
  if isinstance(values, Value):
    values = [values]
  values = [v.mlir_value for v in values]
  transform.YieldOp(values)


@overload
def lower_to_linalg(
    func: Callable[..., Any],
    *args: Any,
    schedule: Schedule | None = None,
    dump_ir: bool = False,
    dump_schedule: bool = False,
) -> ir.Module:
  ...


@overload
def lower_to_linalg(
    module: ir.Module,
    *,
    schedule: Schedule | None = None,
    dump_ir: bool = False,
    dump_schedule: bool = False,
) -> ir.Module:
  ...


def lower_to_linalg(
    func_or_module: Union[Callable[..., Any], ir.Module],
    *args: Any,
    schedule: Schedule | None = None,
    dump_ir: bool = False,
    dump_schedule: bool = False,
) -> ir.Module:
  """Lowers a function to linalg IR and applies a JASC schedule.

  Args:
    func_or_module: A JAX function or an MLIR module to be lowered to linalg IR.
    *args: Arguments that will be passed to the JAX function.
    schedule: The schedule to apply at linalg level
    dump_ir: Whether to dump the transformed IR after each pass
    dump_schedule: Whether to dump the schedule after creation

  Returns:
    An MLIR module with linalg IR with similar semantics to func
  """
  if isinstance(func_or_module, ir.Module):
    module = func_or_module
  else:
    module = lower_to_stablehlo(func_or_module, *args)
  insert_schedule(module, schedule, dump_schedule)
  call_kernel.lower_to_linalg(module, dump_ir)
  return module


def lower_to_stablehlo(
    func: Callable[..., Any],
    *args: Any,
) -> ir.Module:
  """Lowers a function to StableHLO IR.

  Args:
    func: To function to be lowered to linalg IR.
    *args: Arguments that will be passed to func.

  Returns:
    An MLIR module with StableHLO IR with similar semantics to func.
  """

  with primitives.enable_jasc_lowering():
    ir_module = jax.jit(func).lower(*args).compiler_ir("stablehlo")

  # Make sure this lowering is not cached and possibly used by `jax.jit` later.
  jax.clear_caches()
  return ir_module
