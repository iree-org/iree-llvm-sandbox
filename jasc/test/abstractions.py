# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for JASC transform op abstractions."""
from __future__ import annotations

from typing import Callable, Sequence

from absl import app
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import scf
from jaxlib.mlir.dialects import transform
from jaxlib.mlir.dialects.transform import structured

from jasc import jasc
from transform_ops import jasc_transform_ops

tests: list[Callable[[], None]] = []
jasc.set_auto_normalization(False)


def run(f):
  def test():
    print("\nTEST:", f.__name__)
    f()

  tests.append(test)
  return f


def print_schedule(schedule: Callable) -> Callable:
  def decorated() -> None:
    with ir.Context():
      module = ir.Module.parse("")
      jasc.insert_schedule(module, schedule=schedule, dump_schedule=True)
      module.operation.verify()

  decorated.__name__ = schedule.__name__
  return decorated


# CHECK-LABEL: TEST: test_alloca_to_global
@run
@print_schedule
def test_alloca_to_global(program: jasc.OpHandle) -> None:
  get_global, global_ = program.alloca_to_global()
  assert isinstance(get_global, jasc.OpHandle)
  assert isinstance(global_, jasc.OpHandle)
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-DAG:    %[[V0:.*]] = cast
  # CHECK-DAG:    %{{.*}}, %{{.*}} = transform.memref.alloca_to_global %[[V0]]
  # CHECK-SAME:     (!transform.op<"memref.alloca">)
  # CHECK-SAME:     -> (!transform.any_op, !transform.any_op)


# CHECK-LABEL: TEST: test_apply_cse
@run
@print_schedule
def test_apply_cse(program: jasc.OpHandle) -> None:
  program.apply_cse()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   apply_cse to %[[ARG0]] : !transform.any_op


# CHECK-LABEL: TEST: test_apply_dce
@run
@print_schedule
def test_apply_dce(program: jasc.OpHandle) -> None:
  program.apply_dce()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   apply_dce to %[[ARG0]] : !transform.any_op


# CHECK-LABEL: TEST: test_apply_licm_self
@run
@print_schedule
def test_apply_licm_self(program: jasc.OpHandle) -> None:
  program.apply_licm()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:  apply_licm to %[[ARG0]]


# CHECK-LABEL: TEST: test_apply_licm_empty
@run
@print_schedule
def test_apply_licm_empty(program: jasc.OpHandle) -> None:
  program.apply_licm(to=[])
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NOT:   transform.structured.match ops
  # CHECK-NOT:   apply_licm


# CHECK-LABEL: TEST: test_apply_licm_single
@run
@print_schedule
def test_apply_licm_single(program: jasc.OpHandle) -> None:
  program.apply_licm(to=["scf.for"])
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT: %[[VAL0:.*]] = transform.structured.match ops{["scf.for"]}
  # CHECK-SAME: in %[[ARG0]]
  # CHECK-NEXT:   apply_licm to %[[VAL0]]


# CHECK-LABEL: TEST: test_apply_licm_multi
@run
@print_schedule
def test_apply_licm_multi(program: jasc.OpHandle) -> None:
  program.apply_licm(to=["scf.for", "scf.forall"])
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT: %[[VAL0:.*]] = transform.structured.match ops{["scf.for", "scf.forall"]}
  # CHECK-SAME: in %[[ARG0]]
  # CHECK-NEXT:   apply_licm to %[[VAL0]]


# CHECK-LABEL: TEST: test_apply_licm_mixed
@run
@print_schedule
def test_apply_licm_mixed(program: jasc.OpHandle) -> None:
  scf_for = program.match_ops("scf.for")
  program.apply_licm(to=["scf.forall", scf_for])
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-DAG: %[[VL0:.*]] = trans{{.*}}.match ops{["scf.for"]} in %[[ARG0]]
  # CHECK-DAG:   apply_licm to %[[V0]]
  # CHECK-DAG: %[[V1:.*]] = trans{{.*}}.match ops{["scf.forall"]} in %[[ARG0]]
  # CHECK-DAG:   apply_licm to %[[V1]]


# CHECK-LABEL: TEST: test_apply_patterns_empty
@run
@print_schedule
def test_apply_patterns_empty(program: jasc.OpHandle) -> None:
  with program.apply_patterns():
    pass
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   apply_patterns to %[[ARG0]] {
  # CHECK-NEXT:   } : !transform.any_op


# CHECK-LABEL: TEST: test_apply_patterns_args
@run
@print_schedule
def test_apply_patterns_args(program: jasc.OpHandle) -> None:
  with program.apply_patterns(apply_cse=True):
    pass
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   apply_patterns to %[[ARG0]] {
  # CHECK-NEXT:   } {apply_cse} : !transform.any_op


# CHECK-LABEL: TEST: test_apply_patterns_multiple
@run
@print_schedule
def test_apply_patterns_multiple(program: jasc.OpHandle) -> None:
  with program.apply_patterns():
    transform.ApplyCanonicalizationPatternsOp()
    transform.ApplyCanonicalizationPatternsOp()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   apply_patterns to %[[ARG0]] {
  # CHECK-NEXT:     transform.apply_patterns.canonicalization
  # CHECK-NEXT:     transform.apply_patterns.canonicalization
  # CHECK-NEXT:   } : !transform.any_op


# CHECK-LABEL: TEST: test_apply_schedule_in_module
@run
def test_apply_schedule_in_module() -> None:
  def schedule(program: jasc.OpHandle) -> None:
    func = program.match_ops("func.func")
    func.apply_dce()

  with ir.Context():
    module = ir.Module.parse("""
      module {
        func.func @foo() {
            %c0 = arith.constant 0 : i32
            func.return
        }
      }""")
    jasc.insert_schedule(module, schedule, dump_schedule=True)
    jasc.apply_schedule(module, dump_ir=False)
    print(module)
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[VAL0:.*]] = transform.structured.match ops{["func.func"]}
  # CHECK-SAME:     in %[[ARG0]]
  # CHECK-NEXT:   apply_dce to %[[VAL0]]
  # CHECK-NEXT: }
  # CHECK-NEXT: module {
  # CHECK-NEXT:   func.func @foo() {
  # CHECK-NEXT:     return
  # CHECK-NEXT:   }
  # CHECK-NEXT: }


# CHECK-LABEL: TEST: test_apply_schedule_outside_module
@run
def test_apply_schedule_outside_module() -> None:
  def schedule(program: jasc.OpHandle) -> None:
    func = program.match_ops("func.func")
    func.apply_dce()

  with ir.Context():
    module = ir.Module.parse("""
      module {
        func.func @foo() {
            %c0 = arith.constant 0 : i32
            func.return
        }
      }""")
    jasc.apply_schedule(module, schedule, dump_ir=False, dump_schedule=True)
    print(module)
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[VAL0:.*]] = transform.structured.match ops{["func.func"]}
  # CHECK-SAME:     in %[[ARG0]]
  # CHECK-NEXT:   apply_dce to %[[VAL0]]
  # CHECK-NEXT: }
  # CHECK-NEXT: module {
  # CHECK-NEXT:   func.func @foo() {
  # CHECK-NEXT:     return
  # CHECK-NEXT:   }
  # CHECK-NEXT: }


# CHECK-LABEL: TEST: test_apply_tuning_config
@run
@print_schedule
def test_apply_tuning_config(program: jasc.OpHandle) -> None:
  program.apply_tuning_config(
      [16, ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 16)]
  )
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   transform.jasc.apply_tuning_config %[[ARG0]]
  # CHECK-SAME:     {config = [16 : i32, 16 : i32]}


# CHECK-LABEL: TEST: test_buffer_loop_hoisting
@run
@print_schedule
def test_buffer_loop_hoisting(program: jasc.OpHandle) -> None:
  program.buffer_loop_hoisting()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   buffer_loop_hoisting %[[ARG0]]


# CHECK-LABEL: TEST: test_bufferize_to_allocation
@run
@print_schedule
def test_bufferize_to_allocation(op: jasc.OpHandle) -> None:
  result = op.bufferize_to_allocation()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %{{.*}}, %{{.*}} = transform.structured.bufferize_to_allocation
  # CHECK-SAME:     %[[ARG0]] : !transform.any_op
  assert isinstance(result.allocated_buffer, jasc.ValueHandle)
  assert isinstance(result.new_ops, jasc.OpHandle)


# CHECK-LABEL: TEST: test_cast_type
@run
@print_schedule
def test_cast_type(program: jasc.OpHandle):
  program.cast(transform.OperationType.get("test.foo_op"))
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[V0:.*]]: !transform.any_op):
  # CHECK-NEXT: %[[V1:.*]] = cast %[[V0]]
  # CHECK-SAME: !transform.any_op to !transform.op<"test.foo_op">


# CHECK-LABEL: TEST: test_cast_string
@run
@print_schedule
def test_cast_string(program: jasc.OpHandle):
  program.cast("test.foo_op")
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[V0:.*]]: !transform.any_op):
  # CHECK-NEXT: %[[V1:.*]] = cast %[[V0]]
  # CHECK-SAME: !transform.any_op to !transform.op<"test.foo_op">


# CHECK-LABEL: TEST: test_create_async_groups
@run
@print_schedule
def test_create_async_groups(program: jasc.OpHandle):
  program.create_async_groups()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[V0:.*]]: !transform.any_op):
  # CHECK-NEXT: %[[V1:.*]] = transform.nvgpu.create_async_groups %[[V0]]


# CHECK-LABEL: TEST: test_custom_default_value_f32_tuning_param
@run
@print_schedule
def test_custom_default_value_f32_tuning_param(_: jasc.OpHandle) -> None:
  jasc.tuning_param(default_value=ir.FloatAttr.get_f32(1.0))
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[VAL0:.*]] = transform.jasc.tuning_param
  # CHECK-SAME:     {default_value = 1.000000e+00 : f32}


# CHECK-LABEL: TEST: test_custom_default_value_int_tuning_param
@run
@print_schedule
def test_custom_default_value_int_tuning_param(_: jasc.OpHandle) -> None:
  jasc.tuning_param(
      default_value=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 1)
  )
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[VAL0:.*]] = transform.jasc.tuning_param
  # CHECK-SAME:     {default_value = 1 : i32}


# CHECK-LABEL: TEST: generic_tuning_param
@run
@print_schedule
def generic_tuning_param(_: jasc.OpHandle) -> None:
  jasc.tuning_param()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[VAL0:.*]] = transform.jasc.tuning_param
  # CHECK-SAME:     {default_value = 1 : i32}


# CHECK-LABEL: TEST: test_handle_hierarchy
@run
@print_schedule
def test_handle_hierarchy(program: jasc.OpHandle) -> None:
  foo_op_handle = program.match_ops("test.foo_op")
  foo_tag_handle = program.match_tag("foo_tag")
  assert foo_op_handle.parent == program
  assert foo_op_handle in program.children
  assert foo_tag_handle.parent == program
  assert foo_tag_handle in program.children
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[VAL_0:.*]]: !transform.any_op):
  # CHECK-NEXT: %[[VAL_1:.*]] = transform.structured.match ops{["test.foo_op"]}
  # CHECK-SAME:   in %[[VAL_0]]
  # CHECK-SAME:     -> !transform.op<"test.foo_op">
  # CHECK-NEXT: %[[VAL_2:.*]] = transform.structured.match interface{LinalgOp}
  # CHECK-SAME:   in %[[VAL_0]]
  # CHECK-NEXT: transform.jasc.match_tag ["foo_tag"] in %[[VAL_2]]


# CHECK-LABEL: TEST: test_eliminate_empty_tensors
@run
@print_schedule
def test_eliminate_empty_tensors(program: jasc.OpHandle) -> None:
  program.eliminate_empty_tensors()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   transform.bufferization.eliminate_empty_tensors %[[ARG0]] : !transform.any_op


# CHECK-LABEL: TEST: test_fold_fill_into_pad
@run
@print_schedule
def test_fold_fill_into_pad(program: jasc.OpHandle) -> None:
  with program.apply_patterns():
    jasc_transform_ops.ApplyFoldFillIntoPadPatternsOp()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   apply_patterns to %[[ARG0]] {
  # CHECK-NEXT:     transform.apply_patterns.fold_fill_into_pad


# CHECK-LABEL: TEST: test_foreach_empty
@run
@print_schedule
def test_foreach_empty(program: jasc.OpHandle) -> None:
  with program.foreach().body:
    pass
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   foreach %[[ARG0]] : !transform.any_op {
  # CHECK-NEXT:   ^{{.*}}(%[[ARG1:.*]]: !transform.any_op):
  # CHECK-NEXT:   }


# CHECK-LABEL: TEST: test_foreach_only_yield
@run
@print_schedule
def test_foreach_only_yield(program: jasc.OpHandle) -> None:
  with program.foreach().body:
    jasc.yield_()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   foreach %[[ARG0]] : !transform.any_op {
  # CHECK-NEXT:   ^{{.*}}(%[[ARG1:.*]]: !transform.any_op):
  # CHECK-NEXT:   }


# CHECK-LABEL: TEST: test_foreach_simple_noyield
@run
@print_schedule
def test_foreach_simple_noyield(program: jasc.OpHandle) -> None:
  with program.foreach().body as arg:
    assert isinstance(arg, jasc.OpHandle)
    arg.print()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   foreach %[[ARG0]] : !transform.any_op {
  # CHECK-NEXT:   ^{{.*}}(%[[ARG1:.*]]: !transform.any_op):
  # CHECK-NEXT:     transform.print %[[ARG1]] : !transform.any_op
  # CHECK-NEXT:   }


# CHECK-LABEL: TEST: test_foreach_simple_yield
@run
@print_schedule
def test_foreach_simple_yield(program: jasc.OpHandle) -> None:
  with program.foreach().body as arg:
    arg.print()
    jasc.yield_()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   foreach %[[ARG0]] : !transform.any_op {
  # CHECK-NEXT:   ^{{.*}}(%[[ARG1:.*]]: !transform.any_op):
  # CHECK-NEXT:     transform.print %[[ARG1]] : !transform.any_op
  # CHECK-NEXT:   }


# CHECK-LABEL: TEST: test_foreach_explicit_yield
@run
@print_schedule
def test_foreach_explicit_yield(program: jasc.OpHandle) -> None:
  foreach = program.foreach([transform.AnyOpType.get()] * 2)
  with foreach.body as arg:
    jasc.yield_([arg, arg])
  foreach.results[0].print()
  foreach.results[1].print()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[V0:.*]]:2 = foreach %[[ARG0]] : !transform.any_op -> !transform.any_op, !transform.any_op {
  # CHECK-NEXT:   ^{{.*}}(%[[ARG1:.*]]: !transform.any_op):
  # CHECK-NEXT:     transform.yield %[[ARG1]], %[[ARG1]] : !transform.any_op, !transform.any_op
  # CHECK-NEXT:   }
  # CHECK-NEXT:   print %[[V0]]#0
  # CHECK-NEXT:   print %[[V0]]#1


# CHECK-LABEL: TEST: test_foreach_yield_one
@run
@print_schedule
def test_foreach_yield_one(program: jasc.OpHandle) -> None:
  foreach = program.foreach(transform.AnyOpType.get())
  with foreach.body as arg:
    jasc.yield_(arg)
  foreach.results[0].print()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[V0:.*]] = foreach %[[ARG0]] : !transform.any_op -> !transform.any_op {
  # CHECK-NEXT:   ^{{.*}}(%[[ARG1:.*]]: !transform.any_op):
  # CHECK-NEXT:     transform.yield %[[ARG1]] : !transform.any_op
  # CHECK-NEXT:   }
  # CHECK-NEXT:   print %[[V0]]


# CHECK-LABEL: TEST: test_foreach_typed_input
@run
@print_schedule
def test_foreach_typed_input(program: jasc.OpHandle) -> None:
  with program.cast("test.foo_op").foreach().body:
    pass
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[CAST:.*]] = cast %[[ARG0]]
  # CHECK-NEXT:   foreach %[[CAST]] : !transform.op<"test.foo_op"> {
  # CHECK-NEXT:   ^{{.*}}(%[[ARG1:.*]]: !transform.op<"test.foo_op">):
  # CHECK-NEXT:   }


# CHECK-LABEL: TEST: test_fuse_into_standalone
@run
@print_schedule
def test_fuse_into_standalone(op: jasc.OpHandle) -> None:
  other_op = op.match_ops("scf.for")
  op.fuse_into(other_op)
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[V0:.*]] = transform.structured.match
  # CHECK-NEXT:   transform.structured.fuse_into_containing_op
  # CHECK-SAME:     %[[ARG0]] into %[[V0]]
  # CHECK-SAME:     (!transform.any_op, !transform.op<"scf.for">)
  # CHECK-SAME:       -> (!transform.any_op, !transform.any_op)


# CHECK-LABEL: TEST: test_fuse_into_autonormalized
@run
@print_schedule
def test_fuse_into_autonormalized(op: jasc.OpHandle) -> None:
  with jasc.autonormalize():
    other_op = op.match_ops("scf.for")
    op.fuse_into(other_op)
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[V0:.*]] = transform.structured.match
  # CHECK-NEXT:   %[[ARG1:.*]] = get_parent_op %[[ARG0:.*]] {
  # CHECK-SAME:     op_name = "func.func"
  # CHECK-NEXT:   apply_patterns to %[[ARG1]] {
  # CHECK-NEXT:     transform.apply_patterns.linalg.tiling_canonicalization
  # CHECK-NEXT:     transform.apply_patterns.fold_fill_into_pad
  # CHECK-NEXT:     transform.apply_patterns.scf.for_loop_canonicalization
  # CHECK-NEXT:     transform.apply_patterns.canonicalization
  # CHECK-NEXT:   }
  # CHECK-NEXT:   %[[ARG2:.*]] = transform.structured.match ops{["scf.for"]}
  # CHECK-SAME:     in %[[ARG1]]
  # CHECK-NEXT:   apply_licm to %[[ARG2]] : !transform.any_op
  # CHECK-NEXT:   apply_cse to %[[ARG1]] : !transform.any_op
  # CHECK-NEXT:   transform.structured.fuse_into_containing_op
  # CHECK-SAME:     %[[ARG0]] into %[[V0]]
  # CHECK-SAME:     (!transform.any_op, !transform.op<"scf.for">)
  # CHECK-SAME:       -> (!transform.any_op, !transform.any_op)


# CHECK-LABEL: TEST: test_get_parent_op_noargs
@run
@print_schedule
def test_get_parent_op_noargs(program: jasc.OpHandle) -> None:
  program.get_parent_op().print()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[V0:.*]] = get_parent_op %[[ARG0]]
  # CHECK-NOT:      deduplicate
  # CHECK-NOT:      isolated_from_above
  # CHECK-NOT:      op_name
  # CHECK-SAME:     : (!transform.any_op) -> !transform.any_op
  # CHECK-NEXT:   print %[[V0]]


# CHECK-LABEL: TEST: test_get_parent_op_allargs
@run
@print_schedule
def test_get_parent_op_allargs(program: jasc.OpHandle) -> None:
  program.get_parent_op(
      deduplicate=True,
      isolated_from_above=True,
      op_name="test.foo_op",
  )
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[V0:.*]] = get_parent_op %[[ARG0]]
  # CHECK-SAME:     deduplicate
  # CHECK-SAME:     isolated_from_above
  # CHECK-SAME:     op_name = "test.foo_op"


# CHECK-LABEL: TEST: test_get_producer_of_operand
@run
@print_schedule
def test_get_producer_of_operand(program: jasc.OpHandle) -> None:
  program.get_producer_of_operand(operand_number=0)
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   get_producer_of_operand %[[ARG0]][0]


# CHECK-LABEL: TEST: test_hoist_redundant_vector_transfers
@run
@print_schedule
def test_hoist_redundant_vector_transfers(program: jasc.OpHandle) -> None:
  program.hoist_redundant_vector_transfers().print()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[V0:.*]] = transform.structured.hoist_redundant_vector_transfers %[[ARG0]]
  # CHECK-DAG:  print %[[V0]]


# CHECK-LABEL: TEST: test_hoist_pad
@run
@print_schedule
def test_hoist_pad(program: jasc.OpHandle) -> None:
  op = program.hoist_pad(num_loops=1)
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[V0:.*]] = transform.structured.hoist_pad %[[ARG0]]
  # CHECK-SAME: by 1 loops


# CHECK-LABEL: TEST: test_insert_slice_to_copy
@run
@print_schedule
def test_insert_slice_to_copy(program: jasc.OpHandle) -> None:
  program.insert_slice_to_copy()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   transform.structured.insert_slice_to_copy %[[ARG0]]
  # CHECK-SAME:     !transform.op<"linalg.copy">


# CHECK-LABEL: TEST: test_interchange
@run
@print_schedule
def test_interchange(program: jasc.OpHandle) -> None:
  program.interchange([0, 2, 1])
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   transform.structured.interchange %[[ARG0]]
  # CHECK-SAME:     iterator_interchange = [0, 2, 1]
  # CHECK-SAME:     (!transform.any_op) -> !transform.any_op


# CHECK-LABEL: TEST: test_map_forall_to_blocks_noargs
@run
@print_schedule
def test_map_forall_to_blocks_noargs(program: jasc.OpHandle) -> None:
  program.map_forall_to_blocks()
  program.print()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[V0:.*]] = transform.gpu.map_forall_to_blocks %[[ARG0]]
  # CHECK-SAME:     (!transform.any_op) -> !transform.any_op
  # CHECK-NEXT:   print %[[V0:.*]]


# CHECK-LABEL: TEST: test_map_forall_to_blocks_args
@run
@print_schedule
def test_map_forall_to_blocks_args(program: jasc.OpHandle) -> None:
  program.map_forall_to_blocks(grid_dims=[4, 2, 1], generate_gpu_launch=True)
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[V0:.*]] = transform.gpu.map_forall_to_blocks %[[ARG0]]
  # CHECK-SAME:     generate_gpu_launch grid_dims = [4, 2, 1]
  # CHECK-SAME:     (!transform.any_op) -> !transform.any_op


# CHECK-LABEL: TEST: test_map_copy_to_threads
@run
@print_schedule
def test_map_copy_to_threads(program: jasc.OpHandle) -> None:
  result = program.map_copy_to_threads(
      total_num_threads=4, desired_bit_alignment=128
  )
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[V0:.*]] = transform.structured.gpu.map_copy_to_threads %[[ARG0]]
  # CHECK-SAME:     total_num_threads = 4 desired_bit_alignment = 128
  # CHECK-SAME:     (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  assert isinstance(result.forall_op, jasc.OpHandle)
  assert isinstance(result.tiled_op, jasc.OpHandle)


# CHECK-LABEL: TEST: test_map_nested_forall_to_threads_noargs
@run
@print_schedule
def test_map_nested_forall_to_threads_noargs(program: jasc.OpHandle) -> None:
  program.map_nested_forall_to_threads().print()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[V0:.*]] = transform.gpu.map_nested_forall_to_threads %[[ARG0]]
  # CHECK-NOT:      sync_after_distribute
  # CHECK-NOT:      warp_size
  # CHECK-SAME:     block_dims = []
  # CHECK-NOT:      sync_after_distribute
  # CHECK-NOT:      warp_size
  # CHECK-SAME:     (!transform.any_op) -> !transform.any_op
  # CHECK:        print %[[V0]]


# CHECK-LABEL: TEST: test_map_nested_forall_to_threads_allargs
@run
@print_schedule
def test_map_nested_forall_to_threads_allargs(program: jasc.OpHandle) -> None:
  program.map_nested_forall_to_threads(
      block_dims=[4, 4], sync_after_distribute=False, warp_size=128
  )
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[V0:.*]] = transform.gpu.map_nested_forall_to_threads %[[ARG0]]
  # CHECK-DAG:      block_dims = [4, 4]
  # CHECK-DAG:      sync_after_distribute = false
  # CHECK-DAG:      warp_size = 128


# CHECK-LABEL: TEST: test_synchronize
@run
@print_schedule
def test_synchronize(program: jasc.OpHandle) -> None:
  barrier = program.synchronize()
  assert isinstance(barrier, jasc.OpHandle)
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   transform.jasc.synchronize %[[ARG0]]
  # CHECK-SAME:     -> !transform.op<"gpu.barrier">


# CHECK-LABEL: TEST: test_vectorize_static
@run
@print_schedule
def test_vectorize_static(program: jasc.OpHandle) -> None:
  program.vectorize([16, 4])
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   transform.structured.vectorize %[[ARG0]]
  # CHECK-SAME:     vector_sizes [16, 4] : !transform.any_op


# CHECK-LABEL: TEST: test_vectorize_array
@run
@print_schedule
def test_vectorize_array(program: jasc.OpHandle) -> None:
  sizes = ir.Attribute.parse("[16, 4]")
  program.vectorize(sizes)
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   transform.structured.vectorize %[[ARG0]]
  # CHECK-SAME:     vector_sizes [16, 4] : !transform.any_op


# CHECK-LABEL: TEST: test_vectorize_autonormalized
@run
@print_schedule
def test_vectorize_autonormalized(program: jasc.OpHandle) -> None:
  with jasc.autonormalize():
    program.vectorize([16, 4])
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[ARG1:.*]] = get_parent_op %[[ARG0:.*]] {
  # CHECK-SAME:     op_name = "func.func"
  # CHECK-NEXT:   apply_patterns to %[[ARG1]] {
  # CHECK-NEXT:     transform.apply_patterns.linalg.tiling_canonicalization
  # CHECK-NEXT:     transform.apply_patterns.fold_fill_into_pad
  # CHECK-NEXT:     transform.apply_patterns.scf.for_loop_canonicalization
  # CHECK-NEXT:     transform.apply_patterns.canonicalization
  # CHECK-NEXT:   }
  # CHECK-NEXT:   %[[ARG2:.*]] = transform.structured.match ops{["scf.for"]}
  # CHECK-SAME:     in %[[ARG1]]
  # CHECK-NEXT:   apply_licm to %[[ARG2]] : !transform.any_op
  # CHECK-NEXT:   apply_cse to %[[ARG1]] : !transform.any_op
  # CHECK-NEXT:   transform.structured.vectorize %[[ARG0]]
  # CHECK-SAME:     vector_sizes [16, 4] : !transform.any_op


# CHECK-LABEL: TEST: test_vectorize_mixed
@run
@print_schedule
def test_vectorize_mixed(program: jasc.OpHandle) -> None:
  sz1 = program.match_ops("arith.constant")
  sz2 = ir.Attribute.parse("4")
  program.vectorize([sz1, sz2])
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[V0:.*]] = transform.structured.match
  # CHECK-NEXT:   transform.structured.vectorize %[[ARG0]]
  # CHECK-SAME:     vector_sizes [%[[V0]] : !transform.op<"arith.constant">, 4]


# CHECK-LABEL: TEST: test_vectorize_scalable
@run
@print_schedule
def test_vectorize_scalable(program: jasc.OpHandle) -> None:
  sz1 = program.match_ops("arith.constant")
  sz2 = ir.Attribute.parse("4")
  program.vectorize([16, [sz1], [sz2], [8]])
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[V0:.*]] = transform.structured.match
  # CHECK-NEXT:   transform.structured.vectorize %[[ARG0]]
  # CHECK-SAME:     vector_sizes [16,
  # CHECK-SAME:       [%[[V0]] : !transform.op<"arith.constant">], [4], [8]]


# CHECK-LABEL: TEST: test_vectorize_args
@run
@print_schedule
def test_vectorize_args(program: jasc.OpHandle) -> None:
  program.vectorize([16, 4], vectorize_nd_extract=True)
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   transform.structured.vectorize %[[ARG0]]
  # CHECK-SAME:     vectorize_nd_extract


# CHECK-LABEL: TEST: test_match_ops_single
@run
@print_schedule
def test_match_ops_single(program: jasc.OpHandle):
  program.match_ops(scf.ForOp)
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[VAL_0:.*]]: !transform.any_op):
  # CHECK-NEXT: %[[VAL_1:.*]] = transform.structured.match ops{["scf.for"]}
  # CHECK-SAME:    in %[[VAL_0]]
  # CHECK-SAME:      -> !transform.op<"scf.for">


# CHECK-LABEL: TEST: test_match_ops_string_name
@run
@print_schedule
def test_match_ops_string_name(program: jasc.OpHandle):
  program.match_ops("linalg.matmul")
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[VAL_0:.*]]: !transform.any_op):
  # CHECK-NEXT: %[[VAL_1:.*]] = transform.structured.match
  # CHECK-SAME:   ops{["linalg.matmul"]} in %[[VAL_0]]


# CHECK-LABEL: TEST: test_match_ops_string_iface
@run
@print_schedule
def test_match_ops_string_iface(program: jasc.OpHandle):
  program.match_ops("LinalgOp")
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[VAL_0:.*]]: !transform.any_op):
  # CHECK-NEXT: %[[VAL_1:.*]] = transform.structured.match
  # CHECK-SAME:   interface{LinalgOp} in %[[VAL_0]]


# CHECK-LABEL: TEST: test_match_ops_iface
@run
@print_schedule
def test_match_ops_iface(program: jasc.OpHandle):
  program.match_ops(structured.MatchInterfaceEnum.LinalgOp)
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[VAL_0:.*]]: !transform.any_op):
  # CHECK-NEXT: %[[VAL_1:.*]] = transform.structured.match
  # CHECK-SAME:   interface{LinalgOp} in %[[VAL_0]]


# CHECK-LABEL: TEST: test_match_ops_multiple
@run
@print_schedule
def test_match_ops_multiple(program: jasc.OpHandle):
  program.match_ops([scf.ForOp, scf.ForallOp])
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[VAL_0:.*]]: !transform.any_op):
  # CHECK-NEXT: %[[VAL_1:.*]] = transform.structured.match
  # CHECK-SAME:   ops{["scf.for", "scf.forall"]} in %[[VAL_0]]
  # CHECK-SAME:     -> !transform.any_op


# CHECK-LABEL: TEST: test_match_ops_mixed
@run
@print_schedule
def test_match_ops_mixed(program: jasc.OpHandle):
  program.match_ops([scf.ForOp, "linalg.matmul", scf.ForallOp])
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[VAL_0:.*]]: !transform.any_op):
  # CHECK-NEXT: %[[VAL_1:.*]] = transform.structured.match
  # CHECK-SAME:   ops{["scf.for", "linalg.matmul", "scf.forall"]} in %[[VAL_0]]
  # CHECK-SAME:     -> !transform.any_op


# CHECK-LABEL: TEST: test_match_tag_single
@run
@print_schedule
def test_match_tag_single(program: jasc.OpHandle):
  program.match_tag("foo_tag")
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[VAL_0:.*]]: !transform.any_op):
  # CHECK-NEXT: %[[VAL_1:.*]] = transform.structured.match interface{LinalgOp}
  # CHECK-SAME:   in %[[VAL_0]]
  # CHECK-NEXT: transform.jasc.match_tag ["foo_tag"] in %[[VAL_1]]


# CHECK-LABEL: TEST: test_match_tag_multiple
@run
@print_schedule
def test_match_tag_multiple(program: jasc.OpHandle):
  program.match_tag(["foo_tag_0", "foo_tag_1"])
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[VAL_0:.*]]: !transform.any_op):
  # CHECK-NEXT: %[[VAL_1:.*]] = transform.structured.match interface{LinalgOp}
  # CHECK-SAME:   in %[[VAL_0]]
  # CHECK-NEXT: transform.jasc.match_tag ["foo_tag_0", "foo_tag_1"]
  # CHECK-SAME:   in %[[VAL_1]]


# CHECK-LABEL: TEST: test_one_shot_bufferize_noargs
@run
@print_schedule
def test_one_shot_bufferize_noargs(program: jasc.OpHandle) -> None:
  program.one_shot_bufferize()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[V0:.*]] = transform.bufferization.one_shot_bufferize %[[ARG0]]
  # CHECK-SAME:     (!transform.any_op) -> !transform.any_op


# CHECK-LABEL: TEST: test_one_shot_bufferize_args
@run
@print_schedule
def test_one_shot_bufferize_args(program: jasc.OpHandle) -> None:
  program.one_shot_bufferize(
      allow_return_allocs_from_loops=True,
      allow_unknown_ops=True,
      bufferize_function_boundaries=True,
      function_boundary_type_conversion="IdentityLayoutMap",
      memcpy_op="linalg.copy",
      print_conflicts=True,
      test_analysis_only=True,
  )
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[V0:.*]] = transform.bufferization.one_shot_bufferize
  # CHECK-SAME:     layout{IdentityLayoutMap}
  # CHECK-SAME:     %[[ARG0]]
  # CHECK-SAME:     allow_return_allocs_from_loops = true
  # CHECK-SAME:     allow_unknown_ops = true
  # CHECK-SAME:     bufferize_function_boundaries = true
  # CHECK-SAME:     memcpy_op = "linalg.copy"
  # CHECK-SAME:     print_conflicts = true
  # CHECK-SAME:     test_analysis_only = true
  # CHECK-SAME:     (!transform.any_op) -> !transform.any_op


# CHECK-LABEL: TEST: test_pad_allargs
# TODO(ingomueller): I think that `padding_values` and `pad_to_multiple_of`
#                    should be optional but the mix-in currently doesn't support
#                    that. Add tests once it does.
@run
@print_schedule
def test_pad_allargs(program: jasc.OpHandle):
  result = program.pad(
      padding_values=[0.0, 0.0, 0.0, 0.0],
      padding_dimensions=[0, 1, 2, 3],
      pack_paddings=[1, 1, 1, 1],
      pad_to_multiple_of=[1, 1, 1, 1],
      transpose_paddings=[[0, 1, 2, 3]],
      copy_back_op=jasc.PadCopyBackOp.NONE,
  )
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[VAL_0:.*]]: !transform.any_op):
  # CHECK-NEXT: transform.structured.pad %[[VAL_0]]
  # CHECK-SAME: copy_back_op = "none"
  # CHECK-SAME: pack_paddings = [1, 1, 1, 1]
  # CHECK-SAME: pad_to_multiple_of = [1, 1, 1, 1]
  # CHECK-SAME: padding_dimensions = [0, 1, 2, 3]
  # CHECK-SAME: padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32,
  # CHECK-SAME: 0.000000e+00 : f32, 0.000000e+00 : f32]
  # CHECK-SAME{LITERAL}: transpose_paddings = [[0, 1, 2, 3]]
  assert isinstance(result.pad, jasc.OpHandle)
  assert isinstance(result.padded, jasc.OpHandle)
  assert isinstance(result.copy, jasc.OpHandle)


# CHECK-LABEL: TEST: test_print
@run
@print_schedule
def test_print(program: jasc.OpHandle):
  program.print("debugMessage")
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[VAL_0:.*]]: !transform.any_op):
  # CHECK-NEXT: print %[[VAL_0]] {name = "debugMessage"}


# CHECK-LABEL: TEST: test_select
@run
@print_schedule
def test_select(program: jasc.OpHandle):
  program.select(op_name="test.op")
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[VAL_0:.*]]: !transform.any_op):
  # CHECK-NEXT: select "test.op" in %[[VAL_0]]


# CHECK-LABEL: TEST: test_replace_with_alloc_tensor_explicit_cast
@run
@print_schedule
def test_replace_with_alloc_tensor_explicit_cast(program: jasc.OpHandle):
  program.cast("tensor.empty").replace_with_alloc_tensor()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[V0:.*]]: !transform.any_op):
  # CHECK-NEXT: %[[V1:.*]] = cast %[[V0]] : {{.*}} to !transform.op<"tensor.empty">
  # CHECK-NEXT: transform.bufferization.empty_tensor_to_alloc_tensor %[[V1]]


# CHECK-LABEL: TEST: test_replace_with_alloc_tensor_implicit_cast
@run
@print_schedule
def test_replace_with_alloc_tensor_implicit_cast(program: jasc.OpHandle):
  program.replace_with_alloc_tensor()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[V0:.*]]: !transform.any_op):
  # CHECK-NEXT: %[[V1:.*]] = cast %[[V0]] : {{.*}} to !transform.op<"tensor.empty">
  # CHECK-NEXT: transform.bufferization.empty_tensor_to_alloc_tensor %[[V1]]


# CHECK-LABEL: TEST: test_rewrite_in_destination_passing_style
@run
@print_schedule
def test_rewrite_in_destination_passing_style(program: jasc.OpHandle):
  program.rewrite_in_destination_passing_style()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[VAL_0:.*]]: !transform.any_op):
  # CHECK-NEXT: transform.structured.rewrite_in_destination_passing_style
  # CHECK-SAME: %[[VAL_0]]


# CHECK-LABEL: TEST: test_take_assumed_branch_standalone
@run
@print_schedule
def test_take_assumed_branch_standalone(program: jasc.OpHandle):
  program.take_assumed_branch(take_else_branch=True)
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG_0:.*]]: !transform.any_op):
  # CHECK-NEXT: transform.scf.take_assumed_branch %[[ARG_0]] take_else_branch


# CHECK-LABEL: TEST: test_take_assumed_branch_autonormalized
@run
@print_schedule
def test_take_assumed_branch_autonormalized(program: jasc.OpHandle):
  with jasc.autonormalize():
    program.take_assumed_branch(take_else_branch=True)
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG_0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[ARG1:.*]] = get_parent_op %[[ARG0:.*]] {
  # CHECK-SAME:     op_name = "func.func"
  # CHECK-NEXT:   apply_patterns to %[[ARG1]] {
  # CHECK-NEXT:     transform.apply_patterns.linalg.tiling_canonicalization
  # CHECK-NEXT:     transform.apply_patterns.fold_fill_into_pad
  # CHECK-NEXT:     transform.apply_patterns.scf.for_loop_canonicalization
  # CHECK-NEXT:     transform.apply_patterns.canonicalization
  # CHECK-NEXT:   }
  # CHECK-NEXT:   %[[ARG2:.*]] = transform.structured.match ops{["scf.for"]}
  # CHECK-SAME:     in %[[ARG1]]
  # CHECK-NEXT:   apply_licm to %[[ARG2]] : !transform.any_op
  # CHECK-NEXT:   apply_cse to %[[ARG1]] : !transform.any_op
  # CHECK-NEXT: transform.scf.take_assumed_branch %[[ARG_0]] take_else_branch


# CHECK-LABEL: TEST: test_tile_to_for_sizes
@run
@print_schedule
def test_tile_to_for_sizes(program: jasc.OpHandle):
  result = program.tile(loop=jasc.TileLoopKind.FOR, tile_sizes=(2, 4))
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[VAL_0:.*]]: !transform.any_op):
  # CHECK-NEXT: transform.structured.tile_using_for %[[VAL_0]][2, 4]
  assert isinstance(result.tiled_op, jasc.OpHandle)
  for loop in result.loops:
    assert isinstance(loop, jasc.OpHandle)


# CHECK-LABEL: TEST: test_tile_to_for_parametric
@run
@print_schedule
def test_tile_to_for_parametric(program: jasc.OpHandle) -> None:
  tile_size = jasc.tuning_param(
      default_value=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 1)
  )
  program.tile(loop=jasc.TileLoopKind.FOR, tile_sizes=[tile_size])
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[VAL0:.*]] = transform.jasc.tuning_param
  # CHECK-SAME:     {default_value = 1 : i32}
  # CHECK-NEXT:   transform.structured.tile_using_for %[[ARG0]][%[[VAL0]]]


# CHECK-LABEL: TEST: test_tile_to_forall_string
@run
@print_schedule
def test_tile_to_forall_string(program: jasc.OpHandle):
  result = program.tile(
      loop=jasc.TileLoopKind.FORALL,
      tile_sizes=[64, 64, 1],
      mapping="[#gpu.block<z>, #gpu.block<y>, #gpu.block<x>]",
  )
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[V0:.*]]: !transform.any_op):
  # CHECK-NEXT: transform.structured.tile_using_forall %[[V0]]
  # CHECK-SAME: tile_sizes [64, 64, 1]
  # CHECK-SAME: (mapping = [#gpu.block<z>, #gpu.block<y>, #gpu.block<x>])
  assert len(result.loops) == 1
  assert isinstance(result.loops[0], jasc.OpHandle)
  assert isinstance(result.tiled_op, jasc.OpHandle)


# CHECK-LABEL: TEST: test_tile_to_forall_nomapping
@run
@print_schedule
def test_tile_to_forall_nomapping(program: jasc.OpHandle):
  program.tile(
      loop=jasc.TileLoopKind.FORALL,
      tile_sizes=[64, 64, 1],
  )
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[V0:.*]]: !transform.any_op):
  # CHECK-NEXT: transform.structured.tile_using_forall %[[V0]]
  # CHECK-NOT:  mapping


# CHECK-LABEL: TEST: test_tile_to_forall_list
@run
@print_schedule
def test_tile_to_forall_list(program: jasc.OpHandle):
  program.tile(
      loop=jasc.TileLoopKind.FORALL,
      tile_sizes=[64, 64, 1],
      mapping=["#gpu.block<x>", ir.Attribute.parse("#gpu.block<y>")],
  )
  # CHECK: (mapping = [#gpu.block<x>, #gpu.block<y>])


# CHECK-LABEL: TEST: test_tile_to_forall_attr
@run
@print_schedule
def test_tile_to_forall_attr(program: jasc.OpHandle):
  program.tile(
      loop=jasc.TileLoopKind.FORALL,
      tile_sizes=[64, 64, 1],
      mapping=["#gpu.block<x>", ir.Attribute.parse("#gpu.block<y>")],
  )
  # CHECK: (mapping = [#gpu.block<x>, #gpu.block<y>])


# CHECK-LABEL: TEST: test_tile_to_forall_autonormalized
@run
@print_schedule
def test_tile_to_forall_autonormalized(program: jasc.OpHandle):
  with jasc.autonormalize():
    program.tile(
        loop=jasc.TileLoopKind.FORALL,
        tile_sizes=[64, 64, 1],
        mapping=[],
    )
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[ARG1:.*]] = get_parent_op %[[ARG0:.*]] {
  # CHECK-SAME:     op_name = "func.func"
  # CHECK-NEXT:   apply_patterns to %[[ARG1]] {
  # CHECK-NEXT:     transform.apply_patterns.linalg.tiling_canonicalization
  # CHECK-NEXT:     transform.apply_patterns.fold_fill_into_pad
  # CHECK-NEXT:     transform.apply_patterns.scf.for_loop_canonicalization
  # CHECK-NEXT:     transform.apply_patterns.canonicalization
  # CHECK-NEXT:   }
  # CHECK-NEXT:   %[[ARG2:.*]] = transform.structured.match ops{["scf.for"]}
  # CHECK-SAME:     in %[[ARG1]]
  # CHECK-NEXT:   apply_licm to %[[ARG2]] : !transform.any_op
  # CHECK-NEXT:   apply_cse to %[[ARG1]] : !transform.any_op
  # CHECK-NEXT: transform.structured.tile_using_forall %[[ARG0]]
  # CHECK-SAME:   tile_sizes [64, 64, 1]
  # CHECK-SAME:   (mapping = [])


# CHECK-LABEL: TEST: test_vectorize_children_and_apply_patterns
@run
@print_schedule
def test_vectorize_children_and_apply_patterns(program: jasc.OpHandle) -> None:
  program.vectorize_children_and_apply_patterns(
      disable_multi_reduction_to_contract_patterns=True,
      disable_transfer_permutation_map_lowering_patterns=True,
      vectorize_nd_extract=True,
      vectorize_padding=True,
  )
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   transform.structured.vectorize_children_and_apply_patterns %[[ARG0]] {
  # CHECK-SAME:   disable_multi_reduction_to_contract_patterns
  # CHECK-SAME:   disable_transfer_permutation_map_lowering_patterns
  # CHECK-SAME:   vectorize_nd_extract
  # CHECK-SAME:   vectorize_padding}


# CHECK-LABEL: TEST: test_match_sparse_inout
@run
@print_schedule
def test_match_sparse_inout(program: jasc.OpHandle):
  program.match_sparse_inout_ops()
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:  transform.sparse_tensor.match.sparse_inout %[[ARG0]]


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  for test_fun in tests:
    test_fun()


if __name__ == "__main__":
  app.run(main)
