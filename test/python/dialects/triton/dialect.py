# RUN: %PYTHON %s | FileCheck %s

from mlir_structured.dialects import triton as tt
from mlir_structured.dialects import triton_gpu as ttg
from mlir_structured.ir import (
    Context,
    IntegerType,
    Location,
    RankedTensorType,
)


def run(f):
  print("\nTEST:", f.__name__)
  with Context(), Location.unknown():
    tt.register_dialect()
    ttg.register_dialect()
    f()
  return f


# CHECK-LABEL: TEST: testTritionDialectRegistered
@run
def testTritionDialectRegistered():
  i32 = IntegerType.get_signless(32)
  tensor_type = RankedTensorType.get([16], i32)
  r = tt.MakeRangeOp(tensor_type, 0, 16)
  # CHECK-NEXT: %{{.*}} = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  print(r)


# CHECK-LABEL: TEST: testTritionGpuDialectRegistered
@run
def testTritionGpuDialectRegistered():
  i32 = IntegerType.get_signless(32)
  tensor_type = RankedTensorType.get([16], i32)
  # CHECK-NEXT: %{{.*}} = "triton_gpu.alloc_tensor"() : () -> tensor<16xi32>
  t = ttg.AllocTensorOp(tensor_type)
  print(t)
