#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys, time, os

from typing import Any, List, Mapping, Optional, Sequence
import numpy as np
import argparse

from mlir.ir import *
from mlir.dialects import arith, func, linalg, tensor, scf, func, memref
from mlir.dialects.linalg.opdsl.lang import *

from examples.core.problem_definition import *


def save_mlir(mlir_txt, dest):
  f = open(dest, "w")
  f.write(mlir_txt)
  f.close()


def attach_inplaceable_attributes(func: func.FuncOp,
                                  inplaceable: Sequence[Optional[bool]]):

  # Create the following affine_map
  # (d0, d1)[s0, s1] -> (s0 + d0*s1 + d1)
  d0 = AffineDimExpr.get(0)
  d1 = AffineDimExpr.get(1)
  s0 = AffineSymbolExpr.get(0)
  s1 = AffineSymbolExpr.get(1)
  mul = AffineMulExpr.get(d0, s1)
  add = AffineAddExpr.get(s0, mul)
  add = AffineAddExpr.get(add, d1)
  map0 = AffineMap.get(2, 2, [add])

  # Add the attributes to the inputs
  attrs = []
  for flag in inplaceable:
    if flag is None:
      attrs.append(DictAttr.get({}))
      continue
    attrs.append(
        DictAttr.get({
            "bufferization.writable": BoolAttr.get(flag),
            "bufferization.buffer_layout": AffineMapAttr.get(map0),
        }))
  func.arg_attrs = attrs


# C = tr(A)*B
@linalg_structured_op
def matmul_TN(
    A=TensorDef(TV.T1, S.K, S.M),
    B=TensorDef(TV.T2, S.K, S.N),
    C=TensorDef(U, S.M, S.N, output=True),
):

  domain(D.m, D.n, D.k)
  implements(ContractionOpInterface)
  C[D.m, D.n] += TypeFn.cast_signed(U, A[D.k, D.m]) * TypeFn.cast_signed(U, B[D.k, D.n])


# C = A*B
@linalg_structured_op
def matmul_NN(
    A=TensorDef(TV.T1, S.M, S.K),
    B=TensorDef(TV.T2, S.K, S.N),
    C=TensorDef(U, S.M, S.N, output=True),
):

  domain(D.m, D.n, D.k)
  implements(ContractionOpInterface)
  C[D.m, D.n] += TypeFn.cast_signed(U, A[D.m, D.k]) * TypeFn.cast_signed(U, B[D.k, D.n])


class GEMM(ProblemDefinition):
  """Problem definition for a fill + matmul + generic op."""

  def __init__(self, trA):
    self.trA = trA

  def shapes_builder(self, sizes: Mapping[str, Any]) -> List[List[int]]:
    """Shape builder function.

        Given a mapping between dimension names / op attributes and their numeric
        values, return the list of lists of shapes of the FuncOp operands. The
        FuncOp is responsible for distinguishing between input operands and results.
        """
    M, N, K = sizes["M"], sizes["N"], sizes["K"]
    if self.trA:
      return [
          [K, M],
          [K, N],
          [M, N],
      ]
    else:
      return [
          [M, K],
          [K, N],
          [M, N],
      ]

  def types_mlir_builder(self, sizes: Mapping[str, Any],
                         types: Sequence[Type]) -> List[Type]:
    shapes = self.shapes_builder(sizes)
    return [
        RankedTensorType.get(s, t) for s, t in zip(shapes,
                                                   list(types) + [types[-1]])
    ]

  def build_problem_under_context_manager(
      self, name: str, types: Sequence[Type]) -> func.FuncOp:

    # Actual benchmarked function called under entry_point_name.
    bench = func.FuncOp(name, (types, [types[-1]]))

    attach_inplaceable_attributes(bench, inplaceable=[False, False, True])

    with InsertionPoint(bench.add_entry_block()):
      if self.trA:
        matmul = matmul_TN(bench.arguments[0],
                           bench.arguments[1],
                           outs=[bench.arguments[2]])
      else:
        matmul = matmul_NN(bench.arguments[0],
                           bench.arguments[1],
                           outs=[bench.arguments[2]])
      func.ReturnOp([matmul])

    return bench


def generate_mlir(func_name, trA, size, dest):
  # Build MLIR GEMM
  with Context() as ctx, Location.unknown() as loc:
    f32 = F32Type.get()
    problem_definition = GEMM(trA)
    mlir_module = Module.create()
    problem_sizes = {"M": size[0], "N": size[1], "K": size[2]}
    types = problem_definition.types_mlir_builder(
        problem_sizes,
        [f32, f32, f32],
    )
    with InsertionPoint(mlir_module.body):
      func = problem_definition.build_problem_under_context_manager(
          func_name, types)
  save_mlir(str(mlir_module), dest)


def main(argv):
  parser = argparse.ArgumentParser("gemm")
  # Paths and naming
  parser.add_argument("--func-name", default="gemm")
  parser.add_argument("--output", default="")

  # Problem description
  parser.add_argument("--M", type=int)
  parser.add_argument("--N", type=int)
  parser.add_argument("--K", type=int)
  parser.add_argument("--trA", action="store_true")
  parser.add_argument("--reps", type=int, default=1)
  args = parser.parse_args()

  # Generate the problem
  M = args.M if args.M else -1
  N = args.N if args.N else -1
  K = args.K if args.K else -1

  size = [M, N, K]
  dest = args.output if args.output else args.func_name + ".mlir"
  generate_mlir(args.func_name, args.trA, size, dest)


if __name__ == "__main__":
  main(os.sys.argv[1:])
