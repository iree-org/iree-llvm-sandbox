# pytype: skip-file

import os, sys, time
from collections.abc import Callable

import numpy as np

from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import std
from mlir.dialects.sparse_tensor import *
from mlir.execution_engine import *
from mlir.runtime import *

from .experts import *
from .transforms import *

from mlir.dialects.linalg.opdsl.lang import *


@linalg_structured_op
def matmul_dsl(
    A=TensorDef(T, S.M, S.K),
    B=TensorDef(T, S.K, S.N),
    C=TensorDef(T, S.M, S.N, output=True)):
  C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


def op_boilerplate(func_name: str, a1: EncodingAttr, a2: EncodingAttr):
  """Returns boilerplate main method.

  This method sets up a boilerplate main method that calls the generated sparse
  kernel. For convenience, this part is purely done as string input. The main
  method takes a dense matrix argument, and prepares two sparse tensors with
  formats 'a1' and 'a2' and initialized from filenames found through environment
  variables. It then calls the kernel, effectively computing C += A B.
  """
  return f"""
func @main(%c: tensor<8x8xf64>) -> tensor<8x8xf64>
  attributes {{ llvm.emit_c_interface }} {{
  %d0 = constant 0.0: f64
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %fa = call @getTensorFilename(%c0) : (index) -> !llvm.ptr<i8>
  %fb = call @getTensorFilename(%c1) : (index) -> !llvm.ptr<i8>
  %a = sparse_tensor.new %fa : !llvm.ptr<i8> to tensor<8x8xf64,{a1}>
  %b = sparse_tensor.new %fb : !llvm.ptr<i8> to tensor<8x8xf64,{a2}>
  %0 = call @{func_name}(%a, %b, %c) : (tensor<8x8xf64,{a1}>,
                                        tensor<8x8xf64,{a2}>,
                                        tensor<8x8xf64>) -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}}
func private @getTensorFilename(index) -> !llvm.ptr<i8>
"""


def build_sparse_under_context_manager(transform: Callable, a1: EncodingAttr,
                                       a2: EncodingAttr):
  """Build linalg op.

  This method generates a linalg op with for matrix multiplication using
  just the Python API. Effectively, a generic linalg op is constructed
  that computes C(i,j) += A(i,k) * B(k,j) for annotated A and B.
  """
  module = Module.create()
  f64 = F64Type.get()
  a = RankedTensorType.get([8, 8], f64, a1)
  b = RankedTensorType.get([8, 8], f64, a2)
  c = RankedTensorType.get([8, 8], f64)
  arguments = [a, b, c]
  # TODO: how to put an "linalg.inplaceable" attribute on c argument?

  with InsertionPoint(module.body):

    @builtin.FuncOp.from_py_func(*arguments)
    def SpMxSpM(*args):
      return matmul_dsl(args[0], args[1], outs=[args[2]])

  # JIT compile.
  start = time.time()
  # TDO: stop hacking strings.
  string_stitch_input_ir = True
  transformed_module = transform('main', module,
                                 op_boilerplate('SpMxSpM', a1, a2),
                                 string_stitch_input_ir)
  execution_engine = ExecutionEngine(transformed_module, 0)
  elapsed_compilation_s = time.time() - start

  return transformed_module, execution_engine


def compile_and_callback(transform: Callable, callback: Callable,
                         a1: EncodingAttr, a2: EncodingAttr):
  with Context() as ctx, Location.unknown():
    module, execution_engine = build_sparse_under_context_manager(
        transform, a1, a2)
    callback(module, execution_engine)


def compile_and_test(transform: Callable, a1: EncodingAttr, a2: EncodingAttr):
  # Feed nympy array into MLIR computation.
  # Built-in bufferization uses in-out buffers.
  # TODO: replace with inplace comprehensive bufferization.
  Cin = np.zeros((8, 8), np.float64)
  Cin_memref_ptr = ctypes.pointer(
      ctypes.pointer(get_ranked_memref_descriptor(Cin)))
  Cout = np.zeros((8, 8), np.float64)
  Cout_memref_ptr = ctypes.pointer(
      ctypes.pointer(get_ranked_memref_descriptor(Cout)))

  # Expected result in numpy.
  expected = [[15.58, 0.0, 20.14, 0.0, 0.0, 0.0, 0.0, 16.2],
              [0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 25.0, 61.6, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 36.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 98.8, 49.0, 0.0],
              [72.9, 0.0, 101.83, 0.0, 0.0, 0.0, 0.0, 78.58]]

  def callback(module, execution_engine):
    start = time.time()
    execution_engine.invoke('main', Cout_memref_ptr, Cin_memref_ptr)
    elapsed_run_s = time.time() - start
    # Bring back result of MLIR computation into numpy land.
    Cout = ranked_memref_to_numpy(Cout_memref_ptr[0])
    if np.allclose(Cout, expected):
      pass
    else:
      quit(f'FAILURE')

  compile_and_callback(transform, callback, a1, a2)


def main():
  # Read support library.
  # TODO: setup via execution engine
  support = os.getenv('SUPPORTLIB')
  support_lib = ctypes.CDLL(support, mode=ctypes.RTLD_GLOBAL)

  # Generate and run C += AB for all annotation combinations for A and B
  # (dense/sparse in all dimensions, row- and column-wise access, bit-widths)
  # and various compiler options. Note that we do not exhaustively visit
  # all possibilities to keep testing time reasonable (but the parameters
  # can be changed by hand for more coverage).
  count = 0
  with Context() as ctx:
    level1 = [DimLevelType.dense, DimLevelType.dense]
    level2 = [DimLevelType.dense, DimLevelType.compressed]
    level3 = [DimLevelType.compressed, DimLevelType.dense]
    level4 = [DimLevelType.compressed, DimLevelType.compressed]
    order1 = AffineMap.get_permutation([0, 1])
    order2 = AffineMap.get_permutation([1, 0])
    for levels1 in [level1, level2, level3, level4]:
      for levels2 in [level1, level2, level3, level4]:
        for ordering1 in [order1, order2]:
          for ordering2 in [order1, order2]:
            for pwidth in [0, 32]:
              for iwidth in [0, 32]:
                for p in [0, 1]:
                  for v in [0, 1]:
                    for vl in [1, 16, 64]:
                      if v == 0 and vl > 1:
                        continue
                      if v > 0 and vl == 1:
                        continue
                      attr1 = EncodingAttr.get(levels1, ordering1, pwidth,
                                               iwidth)
                      attr2 = EncodingAttr.get(levels2, ordering2, pwidth,
                                               iwidth)
                      opt = (f'parallelization-strategy={p} '
                             f'vectorization-strategy={v} vl={vl}')
                      compiler = ExpertSparseCompiler(options=opt)
                      compile_and_test(compiler, attr1, attr2)
                      count = count + 1

  print('Done with', count, 'tests')


if __name__ == '__main__':
  main()
