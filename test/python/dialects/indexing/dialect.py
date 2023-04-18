# RUN: %PYTHON %s | FileCheck %s

from mlir_structured.dialects import indexing as idx
from mlir_structured.dialects import func
from mlir_structured.ir import Context, IntegerType, F32Type
from mlir_structured.passmanager import PassManager
from mlir_structured.runtime.util import mlir_mod_ctx


def run(f):
  print("\nTEST:", f.__name__)
  with Context():
    idx.register_dialect()
    f()
  return f


# CHECK-LABEL: TEST: testArithValue
@run
def testArithValue():
  i32 = IntegerType.get_signless(32)
  with mlir_mod_ctx():
    ten = idx.Tensor.empty([10, 10], i32)
    # CHECK: %[[TEN:.*]] = tensor.empty() : tensor<10x10xi32>
    print(ten.owner)
    # CHECK: Tensor(%[[TEN]], tensor<10x10xi32>)
    print(ten)

    v = ten[0, 0]
    # CHECK: %[[EXTRACTED:.*]] = tensor.extract %[[TEN]][%{{.*}}, %{{.*}}] : tensor<10x10xi32>
    print(v.owner)
    # CHECK: Scalar(%[[EXTRACTED]], i32)
    print(v)

    w = v + v
    # CHECK: %[[ADDI:.*]] = arith.addi %[[EXTRACTED]], %[[EXTRACTED]] : i32
    print(w.owner)
    z = w * w
    # CHECK: %[[MULI:.*]] = arith.muli %[[ADDI]], %[[ADDI]] : i32
    print(z.owner)


# CHECK-LABEL: TEST: testTensorType
@run
def testTensorType():
  i32 = IntegerType.get_signless(32)
  with mlir_mod_ctx():
    tt = idx.Tensor[(10, 10), i32]
    # CHECK: tensor<10x10xi32>
    print(tt)

    tt = idx.Tensor[(None, None), i32]
    # CHECK: tensor<?x?xi32>
    print(tt)

    tt = idx.IndexTensorType.get([10, 10])
    # CHECK: tensor<10x10xindex>
    print(tt)


# CHECK-LABEL: TEST: testTensorValue
@run
def testTensorValue():
  i32 = IntegerType.get_signless(32)
  with mlir_mod_ctx() as module:

    @func.FuncOp.from_py_func()
    def test_tensor_value():
      ten = idx.Tensor.empty((10, 10), i32)
      # CHECK: Tensor(%[[TEN:.*]], tensor<10x10xi32>)
      print(ten)

      twenty = ten + ten
      # CHECK: %[[ADD:.*]] = "arith.addi"(%[[TEN]], %[[TEN]]) : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<10x10xi32>
      print(twenty.owner)

      one_hundred = ten * ten
      # CHECK: %[[MUL:.*]] = "arith.muli"(%[[TEN]], %[[TEN]]) : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<10x10xi32>
      print(one_hundred.owner)

      return one_hundred

  # CHECK: module {
  # CHECK:   func.func @test_tensor_value() -> tensor<10x10xi32> {
  # CHECK:     %[[TEN]] = tensor.empty() : tensor<10x10xi32>
  # CHECK:     %[[ADD]] = arith.addi %[[TEN]], %[[TEN]] : tensor<10x10xi32>
  # CHECK:     %[[MUL]] = arith.muli %[[TEN]], %[[TEN]] : tensor<10x10xi32>
  # CHECK:     return %[[MUL]] : tensor<10x10xi32>
  # CHECK:   }
  # CHECK: }
  print(module)

  pm = PassManager.parse(
      'builtin.module(func.func(convert-elementwise-to-linalg))')
  pm.run(module.operation)

  # CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
  # CHECK: module {
  # CHECK:   func.func @test_tensor_value() -> tensor<10x10xi32> {
  # CHECK:     %{{.*}} = tensor.empty()
  # CHECK:     %{{.*}} = linalg.generic
  # CHECK:     ^bb0(%{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32):
  # CHECK:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
  # CHECK:       linalg.yield %{{.*}} : i32
  # CHECK:     } -> tensor<10x10xi32>
  # CHECK:     %{{.*}} = linalg.generic
  # CHECK:     ^bb0(%{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32):
  # CHECK:       %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
  # CHECK:       linalg.yield %{{.*}} : i32
  # CHECK:     } -> tensor<10x10xi32>
  # CHECK:     return %{{.*}} : tensor<10x10xi32>
  # CHECK:   }
  # CHECK: }
  print(module)
