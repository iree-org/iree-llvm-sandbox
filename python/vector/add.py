import sys, time
from collections.abc import Callable
from typing import Any, List, NewType, Optional, Sequence, Type

import numpy as np

from mlir.ir import *
from mlir.dialects import arith, builtin, linalg, memref, scf, std, vector
from mlir.execution_engine import *
from mlir.runtime import *

from ..core.experts import *
from ..core.harness import *
from ..core.utils import *
from ..core.compilation import compile_to_execution_engine


def emit_func(name: str, operand_types: Sequence[Type],
              result_types: Sequence[Type]):
  # Actual benchmarked function called under entry_point_name.
  func = builtin.FuncOp(name, (operand_types, result_types))

  vec_type = VectorType(operand_types[0].element_type)
  scal_type = vec_type.element_type
  add = arith.AddIOp if IntegerType.isinstance(scal_type) else arith.AddFOp
  with InsertionPoint(func.add_entry_block()):
    A, B, C = func.arguments
    va, vb = memref.LoadOp(A, []), memref.LoadOp(B, [])
    vc = add(va, vb)
    memref.StoreOp(vc, C, [])
    std.ReturnOp([])

  return func


def create_vector_add(module, name: str, sizes: Sequence[Type], element_type):
  with InsertionPoint(module.body):
    memref_type = MemRefType.get([], VectorType.get(sizes, element_type))
    func = emit_func(name, [memref_type, memref_type, memref_type], [])

  return module


class Transform(TransformationList):

  def __init__(self, **kwargs):
    extra_transforms = [LowerVectors(stage=0), LowerToLLVM()]
    t = extra_transforms if 'transforms' not in kwargs else kwargs[
        'transforms'] + extra_transforms
    d = {'transforms': t}
    kwargs.update(d)
    TransformationList.__init__(self, **kwargs)


def main():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    i8, f32 = IntegerType.get_signless(8), F32Type.get()
    create_vector_add(module, 'add2d_f32', [8, 128], f32)
    create_vector_add(module, 'add3d_i8', [8, 128, 4], i8)

    transform = Transform(
        # print_ir_after_all=True,
        # print_llvmir=True
    )

    def apply_transform_to_entry_point_name(module):
      return transform('add2d_f32', module)

    transformed_module, execution_engine = compile_to_execution_engine(
        module, apply_transform_to_entry_point_name)


if __name__ == '__main__':
  main()
