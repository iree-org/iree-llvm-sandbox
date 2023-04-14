# RUN: %PYTHON %s | FileCheck %s

from mlir_indexing.dialects import indexing as it
from mlir_indexing.dialects import tensor
from mlir_indexing.ir import Context, IntegerType


def run(f):
  print("\nTEST:", f.__name__)
  with Context() as ctx:
    it.register_dialect(ctx)
    f()
  return f


# CHECK-LABEL: TEST: testGatherOp
@run
def testGatherOp():
  with it.mlir_mod_ctx() as module:
    I32 = IntegerType.get_signless(32)
    source = tensor.EmptyOp([10, 10], I32)
    indices = tensor.EmptyOp([1, 2, 2], I32)
    f = it.GatherOp(source, indices, [0, 1])

  # CHECK: module {
  # CHECK:   %0 = tensor.empty() : tensor<10x10xi32>
  # CHECK:   %1 = tensor.empty() : tensor<1x2x2xi32>
  # CHECK:   %2 = indexing.gather %0[%1] coordinates = [0, 1] : (tensor<10x10xi32>, tensor<1x2x2xi32>) -> tensor<1x2xi32>
  # CHECK: }
  print(module)
