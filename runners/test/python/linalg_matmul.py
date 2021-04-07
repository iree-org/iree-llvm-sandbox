# RUN: %PYTHON %s 2>&1 | FileCheck %s

# Bootstrap our local extensions first.
# TODO: Requires that both ${LLVM_INSTALL}/python and ./build are on
# PYTHONPATH
import runners

import sys
from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import std
from mlir.passmanager import *
from mlir.execution_engine import *

# Log everything to stderr and flush so that we have a unified stream to match
# errors/info emitted by MLIR to stderr.
def log(*args):
  print(*args, file=sys.stderr)
  sys.stderr.flush()

boilerplate = """
func @main() -> f32 attributes {llvm.emit_c_interface} {
  %v0 = constant 0.0 : f32
  %v1 = constant 1.0 : f32
  %v2 = constant 2.0 : f32

  %iA = linalg.init_tensor [4, 16] : tensor<4x16xf32>
  %iB = linalg.init_tensor [16, 8] : tensor<16x8xf32>
  %iC = linalg.init_tensor [4, 8] : tensor<4x8xf32>
  %A = linalg.fill(%iA, %v1) : tensor<4x16xf32>, f32 -> tensor<4x16xf32>
  %B = linalg.fill(%iB, %v2) : tensor<16x8xf32>, f32 -> tensor<16x8xf32>
  %C = linalg.fill(%iC, %v0) : tensor<4x8xf32>, f32 -> tensor<4x8xf32>

  %res = call @matmul_on_tensors(%A, %B, %C) :
    (tensor<4x16xf32>, tensor<16x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>)

  %c0 = constant 0 : index
  %0 = vector.transfer_read %res[%c0, %c0], %v0 : tensor<4x8xf32>, vector<2x2xf32>
  %e0 = vector.extract %0[0, 0] : vector<2x2xf32>

  // TODO: FFI-based solution to allow testing and printing with python code.
  return %e0 : f32
}
"""

def transform(module):
  import mlir.conversions
  import mlir.dialects.linalg.passes
  import mlir.transforms

  # TODO: Allow cloning functions from one module to another.
  # Atm we have to resort to string concatenation.
  mod = Module.parse(
    str(module.operation.regions[0].blocks[0].operations[0].operation) +
    boilerplate)

  print(mod)
  pm = PassManager.parse("linalg-comprehensive-bufferize-inplace, " +
                          "func(convert-linalg-to-loops, convert-scf-to-std), " +
                          "convert-vector-to-llvm, " +
                          "convert-std-to-llvm")
  pm.run(mod)
  print(mod)

  return mod

def test_builtin():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    with InsertionPoint(module.body):
      @builtin.FuncOp.from_py_func(RankedTensorType.get((4, 16), f32),
                                   RankedTensorType.get((16, 8), f32),
                                   RankedTensorType.get((4, 8), f32))
      def matmul_on_tensors(lhs, rhs, out):
        return linalg.matmul(lhs, rhs, outs=[out])

    execution_engine = ExecutionEngine(transform(module))

test_builtin()
