import sys, time
from collections.abc import Callable

import numpy as np

from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import std
from mlir.dialects import sparse_tensor
from mlir.execution_engine import *
from mlir.runtime import *

from harness import *
from experts import *
from transforms import *

from mlir.dialects.linalg.opdsl.lang import *


@linalg_structured_op
def matmul_dsl(
    A=TensorDef(T, S.M, S.K),
    B=TensorDef(T, S.K, S.N),
    C=TensorDef(T, S.M, S.N, output=True)):
  C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


def op_boilerplate(func_name: str, a1: str, a2: str):
  """Returns boilerplate main method.

  This method sets up a boilerplate main method that calls the generated sparse
  kernel. For convenience, this part is purely done as string input. The main
  method accepts a filename and dense tensor.  It sets up two sparse matrices
  with formats 'a1' and 'a2', but initialized from the same filename.
  It then calls the kernel, effectively computing C += A^2.
  """
  return f"""
func @main(%arg0: !llvm.ptr<i8>, %arg1: tensor<32x32xf64>) -> (tensor<32x32xf64>) {{
  %0 = sparse_tensor.new %arg0 : !llvm.ptr<i8> to tensor<32x32xf64,{a1}>
  %1 = sparse_tensor.new %arg0 : !llvm.ptr<i8> to tensor<32x32xf64,{a2}>
  %2 = call @{func_name}(%0, %1, %arg1) : (
    tensor<32x32xf64,{a1}>,
    tensor<32x32xf64,{a2}>,
    tensor<32x32xf64>) -> tensor<32x32xf64>
  return %2 : tensor<32x32xf64>
}}
"""


def build_sparse_under_context_manager(transform: Callable, a1: str, a2: str):
  """Build linalg op.

  This method generates a linalg op with for matrix multiplication using
  just the Python API. Effectively, a generic linalg op is constructed
  that computes C(i,j) += A(i,k) * B(k,j) for annotated A and B.
  """
  module = Module.create()
  f64 = F64Type.get()
  a = RankedTensorType.get([32, 32], f64, Attribute.parse(a1))
  b = RankedTensorType.get([32, 32], f64, Attribute.parse(a2))
  c = RankedTensorType.get([32, 32], f64)
  arguments = [a, b, c]

  with InsertionPoint(module.body):

    @builtin.FuncOp.from_py_func(*arguments)
    def matmat(*args):
      return matmul_dsl(args[0], args[1], outs=[args[2]])

  # TODO: make this JIT compile
  #       (1) needs bufferization
  #       (2) tensor initialization
  module = transform(module, op_boilerplate('matmat', a1, a2))


def compile_and_test_sparse_matmat(transform: Callable, a1: str, a2: str):
  with Context() as ctx, Location.unknown():
    execution_engine = build_sparse_under_context_manager(transform, a1, a2)


# Generate and run C += AB for all annotation combinations for A and B
# (dense/sparse in all dimensions, row- and column-wise access).
annotation_list = [
    '#sparse_tensor.encoding<{dimLevelType = ["dense",     "dense"     ]}>',
    '#sparse_tensor.encoding<{dimLevelType = ["dense",     "compressed"]}>',
    '#sparse_tensor.encoding<{dimLevelType = ["compressed","dense"     ]}>',
    '#sparse_tensor.encoding<{dimLevelType = ["compressed","compressed"]}>',
    '#sparse_tensor.encoding<{dimLevelType = ["dense",     "dense"     ], '
    'dimOrdering = affine_map<(i,j) -> (j,i)>}>',
    '#sparse_tensor.encoding<{dimLevelType = ["dense",     "compressed"], '
    'dimOrdering = affine_map<(i,j) -> (j,i)>}>',
    '#sparse_tensor.encoding<{dimLevelType = ["compressed","dense"     ], '
    'dimOrdering = affine_map<(i,j) -> (j,i)>}>',
    '#sparse_tensor.encoding<{dimLevelType = ["compressed","compressed"], '
    'dimOrdering = affine_map<(i,j) -> (j,i)>}>'
]
for a1 in annotation_list:
  for a2 in annotation_list:
    compile_and_test_sparse_matmat(expert_sparse_compiler, a1, a2)
