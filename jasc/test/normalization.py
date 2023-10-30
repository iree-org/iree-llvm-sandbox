"""Tests for JASC transform op abstractions."""
from __future__ import annotations

from typing import Callable, Sequence

from absl import app
from jaxlib.mlir import ir

from jasc import jasc

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


# CHECK-LABEL: TEST: test_auto_apply_loop_normalform
@run
@print_schedule
def test_auto_apply_loop_normalform(program: jasc.OpHandle) -> None:
  with jasc.autonormalize():
    program.tile(
        loop=jasc.TileLoopKind.FORALL, tile_sizes=[64, 64, 1], mapping=[]
    )
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[ARG1:.*]] = get_parent_op %[[ARG0]]
  # CHECK-SAME:     {deduplicate, op_name = "func.func"}
  # CHECK-NEXT:   apply_patterns to %[[ARG1]] {
  # CHECK-NEXT:     transform.apply_patterns.linalg.tiling_canonicalization
  # CHECK-NEXT:     transform.apply_patterns.fold_fill_into_pad
  # CHECK-NEXT:     transform.apply_patterns.scf.for_loop_canonicalization
  # CHECK-NEXT:     transform.apply_patterns.canonicalization
  # CHECK-NEXT:   }
  # CHECK-NEXT:   %[[ARG2:.*]] = transform.structured.match ops{["scf.for"]}
  # CHECK-SAME:     in %[[ARG1]]
  # CHECK-NEXT:   apply_licm to %[[ARG2]]
  # CHECK-NEXT:   apply_cse to %[[ARG1]]
  # CHECK-NEXT:   transform.structured.tile_using_forall %[[ARG0]]


# CHECK-LABEL: TEST: test_autonormalize_contextmanager
@run
@print_schedule
def test_autonormalize_contextmanager(program: jasc.OpHandle) -> None:
  # Checks that autonormalization behavior is preserved outside of the
  # contextmanager
  jasc.set_auto_normalization(True)
  with jasc.autonormalize():
    pass
  program.auto_normalize_parent_func(jasc.LoopNormalform)
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   %[[ARG1:.*]] = get_parent_op %[[ARG0]]
  # CHECK-SAME:     {deduplicate, op_name = "func.func"}
  # CHECK-NEXT:   apply_patterns to %[[ARG1]] {
  # CHECK-NEXT:     transform.apply_patterns.linalg.tiling_canonicalization
  # CHECK-NEXT:     transform.apply_patterns.fold_fill_into_pad
  # CHECK-NEXT:     transform.apply_patterns.scf.for_loop_canonicalization
  # CHECK-NEXT:     transform.apply_patterns.canonicalization
  # CHECK-NEXT:   }
  # CHECK-NEXT:   %[[ARG2:.*]] = transform.structured.match ops{["scf.for"]}
  # CHECK-SAME:     in %[[ARG1]]
  # CHECK-NEXT:   apply_licm to %[[ARG2]]
  # CHECK-NEXT:   apply_cse to %[[ARG1]]


# CHECK-LABEL: TEST: test_normalforms_autonormalization_decorator_plain
@jasc.jasc_transform
def foo_abstraction_0(handle: jasc.Value) -> jasc.Value:
  return jasc.Value(handle.mlir_value, parent=handle)


@run
@print_schedule
def test_normalforms_autonormalization_decorator_plain(
    program: jasc.OpHandle,
) -> None:
  with jasc.autonormalize():
    program.normalize(jasc.LoopNormalform)
    assert program.normalform == jasc.LoopNormalform
    # This will conservatively reset the normalform and also propagate the
    # weaker normalform to the parent
    new_handle = foo_abstraction_0(program)
    assert new_handle.normalform == jasc.AnyForm
    assert program.normalform == jasc.AnyForm
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   apply_patterns to %[[ARG0]] {
  # CHECK-NEXT:     transform.apply_patterns.linalg.tiling_canonicalization
  # CHECK-NEXT:     transform.apply_patterns.fold_fill_into_pad
  # CHECK-NEXT:     transform.apply_patterns.scf.for_loop_canonicalization
  # CHECK-NEXT:     transform.apply_patterns.canonicalization
  # CHECK-NEXT:   }
  # CHECK-NEXT:   %[[ARG1:.*]] = transform.structured.match ops{["scf.for"]}
  # CHECK-SAME:     in %[[ARG0]]
  # CHECK-NEXT:   apply_licm to %[[ARG1]]
  # CHECK-NEXT:   apply_cse to %[[ARG0]]


# CHECK-LABEL: TEST: test_normalforms_autonormalization_decorator_called
@jasc.jasc_transform()
def foo_abstraction_1(handle: jasc.Value) -> jasc.Value:
  return jasc.Value(handle.mlir_value, parent=handle)


@run
@print_schedule
def test_normalforms_autonormalization_decorator_called(
    program: jasc.OpHandle,
) -> None:
  with jasc.autonormalize():
    program.normalize(jasc.LoopNormalform)
    assert program.normalform == jasc.LoopNormalform
    # This will conservatively reset the normalform and also propagate the
    # weaker normalform to the parent
    new_handle = foo_abstraction_1(program)
    assert new_handle.normalform == jasc.AnyForm
    assert program.normalform == jasc.AnyForm


# CHECK-LABEL: TEST: test_normalforms_autonormalization_decorator_args_0
@jasc.jasc_transform(enforced_normalform=jasc.LoopNormalform)
def foo_abstraction_2(handle: jasc.Value) -> jasc.Value:
  return jasc.Value(handle.mlir_value, parent=handle)


@run
@print_schedule
def test_normalforms_autonormalization_decorator_args_0(
    program: jasc.OpHandle,
) -> None:
  with jasc.autonormalize():
    program.normalize(jasc.LoopNormalform)
    assert program.normalform == jasc.LoopNormalform
    # This will retain the normalform
    new_handle = foo_abstraction_2(program)
    assert new_handle.normalform == jasc.LoopNormalform
    assert program.normalform == jasc.LoopNormalform


# CHECK-LABEL: TEST: test_normalforms_autonormalization_decorator_args_1
@jasc.jasc_transform(no_propagate=True)
def foo_abstraction_3(handle: jasc.Value) -> jasc.Value:
  return jasc.Value(handle.mlir_value, parent=handle)


@run
@print_schedule
def test_normalforms_autonormalization_decorator_args_1(
    program: jasc.OpHandle,
) -> None:
  with jasc.autonormalize():
    program.normalize(jasc.LoopNormalform)
    assert program.normalform == jasc.LoopNormalform
    # This will change nothing regarding normalforms
    new_handle = foo_abstraction_3(program)
    assert new_handle.normalform == jasc.AnyForm
    assert program.normalform == jasc.LoopNormalform


# CHECK-LABEL: TEST: test_loop_normalform
@run
@print_schedule
def test_loop_normalform(program: jasc.OpHandle) -> None:
  program.normalize(jasc.LoopNormalform)
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   apply_patterns to %[[ARG0]] {
  # CHECK-NEXT:     transform.apply_patterns.linalg.tiling_canonicalization
  # CHECK-NEXT:     transform.apply_patterns.fold_fill_into_pad
  # CHECK-NEXT:     transform.apply_patterns.scf.for_loop_canonicalization
  # CHECK-NEXT:     transform.apply_patterns.canonicalization
  # CHECK-NEXT:   }
  # CHECK-NEXT:   %[[ARG1:.*]] = transform.structured.match ops{["scf.for"]}
  # CHECK-SAME:     in %[[ARG0]]
  # CHECK-NEXT:   apply_licm to %[[ARG1]]
  # CHECK-NEXT:   apply_cse to %[[ARG0]]


# CHECK-LABEL: TEST: test_no_duplicated_auto_apply
@run
@print_schedule
def test_no_duplicated_auto_apply(
    program: jasc.OpHandle,
) -> None:
  """Checks autonormalization doesn't trigger when handle is in the correct form."""
  with jasc.autonormalize():
    program.normalize(jasc.LoopNormalform)
    program.tile(
        loop=jasc.TileLoopKind.FORALL, tile_sizes=[64, 64, 1], mapping=[]
    )
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT:   apply_patterns to %[[ARG0]] {
  # CHECK-NEXT:     transform.apply_patterns.linalg.tiling_canonicalization
  # CHECK-NEXT:     transform.apply_patterns.fold_fill_into_pad
  # CHECK-NEXT:     transform.apply_patterns.scf.for_loop_canonicalization
  # CHECK-NEXT:     transform.apply_patterns.canonicalization
  # CHECK-NEXT:   }
  # CHECK-NEXT:   %[[ARG1:.*]] = transform.structured.match ops{["scf.for"]}
  # CHECK-SAME:     in %[[ARG0]]
  # CHECK-NEXT:   apply_licm to %[[ARG1]]
  # CHECK-NEXT:   apply_cse to %[[ARG0]]
  # CHECK-NOT:    apply_patterns
  # CHECK-NOT:      transform.apply_patterns.linalg.tiling_canonicalization
  # CHECK-NOT:      transform.apply_patterns.fold_fill_into_pad
  # CHECK-NOT:      transform.apply_patterns.scf.for_loop_canonicalization
  # CHECK-NOT:      transform.apply_patterns.canonicalization
  # CHECK-NOT:    apply_licm
  # CHECK-NOT:    apply_cse
  # CHECK-NEXT:   transform.structured.tile_using_forall %[[ARG0]]


# CHECK-LABEL: TEST: test_propagation
@run
@print_schedule
def test_propagation(program: jasc.OpHandle):
  nested_op = program.match_ops("test.foo_op")
  program.normalize(jasc.LoopNormalform)
  assert program.normalform == jasc.LoopNormalform
  assert nested_op.normalform == jasc.LoopNormalform
  nested_op.tile(
      loop=jasc.TileLoopKind.FORALL, tile_sizes=[64, 64, 1], mapping=[]
  )
  assert nested_op.normalform == jasc.AnyForm
  assert program.normalform == jasc.AnyForm
  # CHECK: transform.sequence
  # CHECK-NEXT: ^{{.*}}(%[[ARG0:.*]]: !transform.any_op):
  # CHECK-NEXT: %[[ARG1]] = transform.structured.match ops{["test.foo_op"]}
  # CHECK-SAME    in %[[ARG0]]
  # CHECK-NEXT:   apply_patterns to %[[ARG0]] {
  # CHECK-NEXT:     transform.apply_patterns.linalg.tiling_canonicalization
  # CHECK-NEXT:     transform.apply_patterns.fold_fill_into_pad
  # CHECK-NEXT:     transform.apply_patterns.scf.for_loop_canonicalization
  # CHECK-NEXT:     transform.apply_patterns.canonicalization
  # CHECK-NEXT:   }
  # CHECK-NEXT:   %[[ARG2:.*]] = transform.structured.match ops{["scf.for"]}
  # CHECK-SAME:     in %[[ARG0]]
  # CHECK-NEXT:   apply_licm to %[[ARG2]]
  # CHECK-NEXT:   apply_cse to %[[ARG0]]
  # CHECK-NEXT:   transform.structured.tile_using_forall %[[ARG1]]


# CHECK-LABEL: TEST: test_normalize_parent_func_multiple_payloads
@run
def test_normalize_parent_func_multiple_payloads() -> None:
  with ir.Context():
    module = ir.Module.parse("""
        func.func public @main(%arg0: f32, %arg1: tensor<8x2xf32>)
            -> (tensor<8x2xf32>, tensor<8x2xf32>) {
          %0 = linalg.fill ins(%arg0 : f32) outs(%arg1 : tensor<8x2xf32>) -> tensor<8x2xf32>
          %1 = linalg.fill ins(%arg0 : f32) outs(%arg1 : tensor<8x2xf32>) -> tensor<8x2xf32>
          return %0, %1 : tensor<8x2xf32>, tensor<8x2xf32>
        }
        """)

    def schedule(program: jasc.OpHandle) -> None:
      with jasc.autonormalize():
        program.match_ops("linalg.fill").auto_normalize_parent_func(
            jasc.LoopNormalform
        )

    jasc.lower_to_linalg(module, schedule=schedule, dump_schedule=True)
    module.operation.verify()

    # CHECK: transform.sequence
    # CHECK:        %[[V0:.*]] = transform.structured.match
    # CHECK-SAME:     "linalg.fill"
    # CHECK:        %[[V1:.*]] = get_parent_op %[[V0]]
    # CHECK-SAME:     deduplicate
    # CHECK:        transform.structured.match {{.*}} in %[[V1]]


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  for test_fun in tests:
    test_fun()


if __name__ == "__main__":
  app.run(main)
