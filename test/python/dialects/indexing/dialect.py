# RUN: %PYTHON %s | FileCheck %s
from random import random

import numpy as np

from mlir_structured.dialects import func, arith, indexing
from mlir_structured.dialects.indexing import Scalar, Tensor, IndexTensorType, gather
from mlir_structured.ir import Context, IntegerType, F64Type
from mlir_structured.passmanager import PassManager
from mlir_structured.runtime.util import mlir_mod_ctx


def run(f):
  print("\nTEST:", f.__name__)
  with Context():
    indexing.register_dialect()
    f()
  return f


# CHECK-LABEL: TEST: testScalarValue
@run
def testScalarValue():
  f64 = F64Type.get()
  i32 = IntegerType.get_signless(32)
  with mlir_mod_ctx() as module:
    zero_f64 = Scalar(arith.ConstantOp(f64, 0.0).result)
    # CHECK: Scalar(%{{.*}}, f64, 0.0)
    print(zero_f64)
    # CHECK: True
    print(zero_f64.is_constant())
    # CHECK: 0.0
    print(zero_f64.literal_value)

    zero_f64 = Scalar(arith.ConstantOp(f64, 0.0))
    # CHECK: Scalar(%{{.*}}, f64, 0.0)
    print(zero_f64)
    # CHECK: True
    print(zero_f64.is_constant())
    # CHECK: 0.0
    print(zero_f64.literal_value)

    zero_f64 = Scalar(0.0)
    # CHECK: Scalar(%{{.*}}, f64, 0.0)
    print(zero_f64)
    # CHECK: True
    print(zero_f64.is_constant())
    # CHECK: 0.0
    print(zero_f64.literal_value)

    zero_i64 = Scalar(0)
    # CHECK: Scalar(%{{.*}}, i64, 0)
    print(zero_i64)
    # CHECK: True
    print(zero_i64.is_constant())
    # CHECK: 0
    print(zero_i64.literal_value)

    zero_i32 = Scalar(0, dtype=i32)
    # CHECK: Scalar(%{{.*}}, i32, 0)
    print(zero_i32)
    # CHECK: True
    print(zero_i32.is_constant())
    # CHECK: 0
    print(zero_i32.literal_value)

    zero_index = Scalar(0, index=True)
    # CHECK: Scalar(%{{.*}}, index, 0)
    print(zero_index)
    # CHECK: True
    print(zero_index.is_constant())
    # CHECK: 0
    print(zero_index.literal_value)

    one_f64 = Scalar(1.0)
    two_f64 = Scalar(2.0)

    three_f64 = one_f64 + two_f64
    # CHECK: %{{.*}} = arith.constant 3.000000e+00 : f64
    print(three_f64.owner)

    x, y = random(), random()
    x_f64, y_f64 = Scalar(x), Scalar(y)

    z_f64 = x_f64 + y_f64
    # CHECK: True
    print(z_f64.literal_value == x + y)
    # CHECK: True
    print(zero_f64.is_constant())


# CHECK-LABEL: TEST: testTensorType
@run
def testTensorType():
  i32 = IntegerType.get_signless(32)
  with mlir_mod_ctx():
    tt = Tensor[(10, 10), i32]
    # CHECK: tensor<10x10xi32>
    print(tt)

    tt = Tensor[(None, None), i32]
    # CHECK: tensor<?x?xi32>
    print(tt)

    tt = IndexTensorType.get([10, 10])
    # CHECK: tensor<10x10xindex>
    print(tt)


# CHECK-LABEL: TEST: testTensorValue
@run
def testTensorValue():
  i32 = IntegerType.get_signless(32)
  with mlir_mod_ctx() as module:

    ten = Tensor.empty((10, 10), i32)
    # CHECK: Tensor(%[[TEN:.*]], tensor<10x10xi32>)
    print(repr(ten))
    # CHECK: %[[TEN]] = tensor.empty() : tensor<10x10xi32>
    print(ten.owner)
    # CHECK: (10, 10)
    print(ten.shape)
    # CHECK: i32
    print(ten.dtype)
    # CHECK: False
    print(ten.is_constant())
    try:
      print(ten.literal_value)
    except ValueError as e:
      # CHECK: Can't build literal from non-constant Tensor
      print(e)

    sum_ten_1 = ten + ten
    # CHECK: %[[ADD:.*]] = arith.addi %[[TEN]], %[[TEN]] : tensor<10x10xi32>
    print(sum_ten_1.owner)

    prod_ten = ten * ten
    # CHECK: %[[MUL:.*]] = arith.muli %[[TEN]], %[[TEN]] : tensor<10x10xi32>
    print(prod_ten.owner)

    x = np.random.random((10, 10))
    ten_x = Tensor(x)
    # CHECK: Tensor(%[[CST1:.*]], tensor<10x10xf64>, [
    print(ten_x)
    # CHECK: (10, 10)
    print(ten_x.shape)
    # CHECK: f64
    print(ten_x.dtype)
    # CHECK: True
    print(ten_x.is_constant())
    # CHECK: True
    print(np.allclose(ten_x.literal_value, x))

    y = np.random.random((10, 10))
    # CHECK: Tensor(%[[CST2:.*]], tensor<10x10xf64>, [
    ten_y = Tensor(y)
    print(ten_y)
    sum_ten_2 = ten_x + ten_y
    # CHECK: Tensor(%[[CST3:.*]], tensor<10x10xf64>, [
    print(sum_ten_2)
    # CHECK: (10, 10)
    print(sum_ten_2.shape)
    # CHECK: f64
    print(sum_ten_2.dtype)
    # CHECK: True
    print(sum_ten_2.is_constant())
    # CHECK: True
    print(np.allclose(sum_ten_2.literal_value, x + y))

    try:
      Tensor(arith.ConstantOp(i32, 0).result)
    except ValueError as e:
      # CHECK: Cannot cast value to TensorValue (from <mlir_structured._mlir_libs._mlir.ir.OpResult
      print(e)

  # CHECK: module {
  # CHECK:   %[[TEN]] = tensor.empty() : tensor<10x10xi32>
  # CHECK:   %[[ADD]] = arith.addi %[[TEN]], %[[TEN]] : tensor<10x10xi32>
  # CHECK:   %[[MUL]] = arith.muli %[[TEN]], %[[TEN]] : tensor<10x10xi32>
  # CHECK:   %[[CST1]] = arith.constant dense<{{.*}}> : tensor<10x10xf64>
  # CHECK:   %[[CST2]] = arith.constant dense<{{.*}}> : tensor<10x10xf64>
  # CHECK:   %[[CST3]] = arith.constant dense<{{.*}}> : tensor<10x10xf64>
  # CHECK: }
  print(module)

  pm = PassManager.parse('builtin.module(convert-elementwise-to-linalg)')
  pm.run(module.operation)

  # CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
  # CHECK: module {
  # CHECK:   %{{.*}} = tensor.empty()
  # CHECK:   %{{.*}} = linalg.generic
  # CHECK:   ^bb0(%{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32):
  # CHECK:     %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
  # CHECK:     linalg.yield %{{.*}} : i32
  # CHECK:   } -> tensor<10x10xi32>
  # CHECK:   %{{.*}} = linalg.generic
  # CHECK:   ^bb0(%{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32):
  # CHECK:     %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
  # CHECK:     linalg.yield %{{.*}} : i32
  # CHECK:   } -> tensor<10x10xi32>
  # CHECK:   %[[CST1]] = arith.constant dense<{{.*}}> : tensor<10x10xf64>
  # CHECK:   %[[CST2]] = arith.constant dense<{{.*}}> : tensor<10x10xf64>
  # CHECK:   %[[CST3]] = arith.constant dense<{{.*}}> : tensor<10x10xf64>
  # CHECK: }
  print(module)


# CHECK-LABEL: TEST: testConcatenateOp
@run
def testConcatenateOp():
  i32 = IntegerType.get_signless(32)
  with mlir_mod_ctx() as module:

    @func.FuncOp.from_py_func()
    def test_concat_op():
      ten = Tensor.empty((10, 10), i32)
      # CHECK: Tensor(%[[TEN:.*]], tensor<10x10xi32>)
      print(ten)

      concat_ten_first_dim = indexing.ConcatenateOp((ten, ten), 0).result
      # CHECK: %{{.*}} = "indexing.concatenate"(%[[TEN]], %[[TEN]]) {dimension = 0 : i64} : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<20x10xi32>
      print(concat_ten_first_dim.owner)

      concat_ten_second_dim = indexing.ConcatenateOp((ten, ten), 1).result
      # CHECK: %{{.*}} = "indexing.concatenate"(%[[TEN]], %[[TEN]]) {dimension = 1 : i64} : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<10x20xi32>
      print(concat_ten_second_dim.owner)


# import jax.numpy as jnp


# CHECK-LABEL: TEST: testLiteralIndexing
@run
def testLiteralIndexing():
  i32 = IntegerType.get_signless(32)
  with mlir_mod_ctx() as module:

    @func.FuncOp.from_py_func()
    def test_tensor_value():
      ten = Tensor.empty((10, 10, 10, 10), i32)
      # CHECK: Tensor(%[[TEN:.*]], tensor<10x10x10x10xi32>)
      print(ten)

      w1 = ten[0, 0]
      # CHECK: %{{.*}} = "indexing.gather"(%[[TEN]], %{{.*}}) {gather_dims = array<i64: 0, 1>, unique} : (tensor<10x10x10x10xi32>, tensor<2xindex>) -> tensor<10x10xi32>
      print(w1.owner)

      w2 = ten[0, :, ...]
      # CHECK: %{{.*}} = "indexing.gather"(%[[TEN]], %{{.*}}) {gather_dims = array<i64: 0>, unique} : (tensor<10x10x10x10xi32>, tensor<1xindex>) -> tensor<10x10x10xi32>
      print(w2.owner)

      w3 = ten[:, 0, ...]
      # CHECK: %{{.*}} = "indexing.gather"(%[[TEN]], %{{.*}}) {gather_dims = array<i64: 1>, unique} : (tensor<10x10x10x10xi32>, tensor<1xindex>) -> tensor<10x10x10xi32>
      print(w3.owner)

      w4 = ten[:, 0, 0]
      # CHECK: %{{.*}} = "indexing.gather"(%[[TEN]], %{{.*}}) {gather_dims = array<i64: 1, 2>, unique} : (tensor<10x10x10x10xi32>, tensor<2xindex>) -> tensor<10x10xi32>
      print(w4.owner)

      w5 = ten[:, :, 0, 0]
      # CHECK: %{{.*}} = "indexing.gather"(%[[TEN]], %{{.*}}) {gather_dims = array<i64: 2, 3>, unique} : (tensor<10x10x10x10xi32>, tensor<2xindex>) -> tensor<10x10xi32>
      print(w5.owner)

      w6 = ten[0, :, :, 0]
      # CHECK: %{{.*}} = "indexing.gather"(%[[TEN]], %{{.*}}) {gather_dims = array<i64: 0, 3>, unique} : (tensor<10x10x10x10xi32>, tensor<2xindex>) -> tensor<10x10xi32>
      print(w6.owner)

      w7 = ten[:, :, :, :]
      # should return the orig tensor because this is a no-op
      # CHECK: Tensor(%[[TEN]], tensor<10x10x10x10xi32>)
      print(w7)

      w8 = ten[[[0], [1]], :, :, :]
      # CHECK: %{{.*}} = "indexing.gather"(%[[TEN]], %{{.*}}) {gather_dims = array<i64: 0>, unique} : (tensor<10x10x10x10xi32>, tensor<2x1xindex>) -> tensor<2x10x10x10xi32>
      print(w8.owner)

      w9 = ten[[[0], [1]], [[0], [1]], :, :]
      # CHECK: %{{.*}} = "indexing.gather"(%[[TEN]], %{{.*}}) {gather_dims = array<i64: 0, 1>, unique} : (tensor<10x10x10x10xi32>, tensor<2x2xindex>) -> tensor<2x10x10xi32>
      print(w9.owner)

      w10 = ten[[[0], [1]], :, [[0], [1]], :]
      # CHECK: %{{.*}} = "indexing.gather"(%[[TEN]], %{{.*}}) {gather_dims = array<i64: 0, 2>, unique} : (tensor<10x10x10x10xi32>, tensor<2x2xindex>) -> tensor<2x10x10xi32>
      print(w10.owner)

      return w1, w2, w3, w4, w5, w6, w7, w8, w9, w10

  pm = PassManager.parse("builtin.module(func.func(cse))")
  pm.run(module.operation)
  # CHECK: module {
  # CHECK:   func.func @test_tensor_value() -> (tensor<10x10xi32>, tensor<10x10x10xi32>, tensor<10x10x10xi32>, tensor<10x10xi32>, tensor<10x10xi32>, tensor<10x10xi32>, tensor<10x10x10x10xi32>, tensor<2x10x10x10xi32>, tensor<2x10x10xi32>, tensor<2x10x10xi32>) {
  # CHECK:     %[[TEN:.*]] = tensor.empty() : tensor<10x10x10x10xi32>
  # CHECK:     %cst = arith.constant dense<0> : tensor<2xindex>
  # CHECK:     %1 = indexing.gather %[[TEN]][%cst] gather_dims([0, 1]) unique : (tensor<10x10x10x10xi32>, tensor<2xindex>) -> tensor<10x10xi32>
  # CHECK:     %cst_0 = arith.constant dense<0> : tensor<1xindex>
  # CHECK:     %2 = indexing.gather %[[TEN]][%cst_0] gather_dims([0]) unique : (tensor<10x10x10x10xi32>, tensor<1xindex>) -> tensor<10x10x10xi32>
  # CHECK:     %3 = indexing.gather %[[TEN]][%cst_0] gather_dims([1]) unique : (tensor<10x10x10x10xi32>, tensor<1xindex>) -> tensor<10x10x10xi32>
  # CHECK:     %4 = indexing.gather %[[TEN]][%cst] gather_dims([1, 2]) unique : (tensor<10x10x10x10xi32>, tensor<2xindex>) -> tensor<10x10xi32>
  # CHECK:     %5 = indexing.gather %[[TEN]][%cst] gather_dims([2, 3]) unique : (tensor<10x10x10x10xi32>, tensor<2xindex>) -> tensor<10x10xi32>
  # CHECK:     %6 = indexing.gather %[[TEN]][%cst] gather_dims([0, 3]) unique : (tensor<10x10x10x10xi32>, tensor<2xindex>) -> tensor<10x10xi32>
  # CHECK:     %cst_1 = arith.constant dense<{{\[}}[0], [1]]> : tensor<2x1xindex>
  # CHECK:     %7 = indexing.gather %[[TEN]][%cst_1] gather_dims([0]) unique : (tensor<10x10x10x10xi32>, tensor<2x1xindex>) -> tensor<2x10x10x10xi32>
  # CHECK:     %cst_2 = arith.constant dense<{{\[}}[0, 0], [1, 1]]> : tensor<2x2xindex>
  # CHECK:     %8 = indexing.gather %[[TEN]][%cst_2] gather_dims([0, 1]) unique : (tensor<10x10x10x10xi32>, tensor<2x2xindex>) -> tensor<2x10x10xi32>
  # CHECK:     %9 = indexing.gather %[[TEN]][%cst_2] gather_dims([0, 2]) unique : (tensor<10x10x10x10xi32>, tensor<2x2xindex>) -> tensor<2x10x10xi32>
  # CHECK:     return %1, %2, %3, %4, %5, %6, %0, %7, %8, %9 : tensor<10x10xi32>, tensor<10x10x10xi32>, tensor<10x10x10xi32>, tensor<10x10xi32>, tensor<10x10xi32>, tensor<10x10xi32>, tensor<10x10x10x10xi32>, tensor<2x10x10x10xi32>, tensor<2x10x10xi32>, tensor<2x10x10xi32>
  # CHECK:   }
  # CHECK: }
  print(module)


# CHECK-LABEL: TEST: testAdvancedIndexing
@run
def testAdvancedIndexing():
  i32 = IntegerType.get_signless(32)
  with mlir_mod_ctx() as module:

    @func.FuncOp.from_py_func()
    def test_tensor_value():
      ten = Tensor.empty((10, 10, 10, 10), i32)
      # CHECK: Tensor(%[[TEN:.*]], tensor<10x10x10x10xi32>)
      print(ten)

      idx_ten = Tensor(np.random.randint(low=0, high=10, size=(5, 6, 1)),
                       index=True)
      # CHECK: %{{.*}} = "arith.constant"() <{value = dense<{{.*}}> : tensor<5x6x1xindex>}> : () -> tensor<5x6x1xindex>
      print(idx_ten.owner)
      # CHECK: True
      print(np.array(idx_ten.literal_value).dtype == np.int64)

      w1 = ten[idx_ten, ...]
      # CHECK: "indexing.gather"(%[[TEN]], %{{.*}}) {gather_dims = array<i64: 0>} : (tensor<10x10x10x10xi32>, tensor<5x6x1xindex>) -> tensor<5x6x10x10x10xi32>
      print(w1.owner)

      w2 = ten[idx_ten, idx_ten, ...]
      # CHECK: "indexing.gather"(%[[TEN]], %{{.*}}) {gather_dims = array<i64: 0, 1>} : (tensor<10x10x10x10xi32>, tensor<5x6x2xindex>) -> tensor<5x6x10x10xi32>
      print(w2.owner)

      w3 = ten[idx_ten, :, idx_ten]
      # CHECK: "indexing.gather"(%[[TEN]], %{{.*}}) {gather_dims = array<i64: 0, 2>} : (tensor<10x10x10x10xi32>, tensor<5x6x2xindex>) -> tensor<5x6x10x10xi32>
      print(w3.owner)

      w4 = ten[idx_ten, :, idx_ten, idx_ten]
      # CHECK: "indexing.gather"(%[[TEN]], %{{.*}}) {gather_dims = array<i64: 0, 2, 3>} : (tensor<10x10x10x10xi32>, tensor<5x6x3xindex>) -> tensor<5x6x10xi32>
      print(w4.owner)

      idx_ten = Tensor(np.random.randint(low=0, high=10, size=(5, 6, 2)),
                       index=True)
      # CHECK: %{{.*}} = "arith.constant"() <{value = dense<{{.*}}> : tensor<5x6x2xindex>}> : () -> tensor<5x6x2xindex>
      print(idx_ten.owner)

      w5 = gather(ten, idx_ten, [0, 1])
      # CHECK: "indexing.gather"(%[[TEN]], %{{.*}}) {gather_dims = array<i64: 0, 1>} : (tensor<10x10x10x10xi32>, tensor<5x6x2xindex>) -> tensor<5x6x10x10xi32>
      print(w5.owner)

      w6 = gather(ten, idx_ten, [0, 2])
      # CHECK: "indexing.gather"(%[[TEN]], %{{.*}}) {gather_dims = array<i64: 0, 2>} : (tensor<10x10x10x10xi32>, tensor<5x6x2xindex>) -> tensor<5x6x10x10xi32>
      print(w6.owner)

      w7 = ten[idx_ten, ...]
      # CHECK: "indexing.gather"(%[[TEN]], %{{.*}}) {gather_dims = array<i64: 0, 1>} : (tensor<10x10x10x10xi32>, tensor<5x6x2xindex>) -> tensor<5x6x10x10xi32>
      print(w7.owner)

      w8 = ten[:, idx_ten, ...]
      # CHECK: "indexing.gather"(%[[TEN]], %{{.*}}) {gather_dims = array<i64: 1, 2>} : (tensor<10x10x10x10xi32>, tensor<5x6x2xindex>) -> tensor<5x6x10x10xi32>
      print(w8.owner)

      ten = Tensor.empty((10, 10, 10, 10, 10), i32)
      # CHECK: Tensor(%[[TEN2:.*]], tensor<10x10x10x10x10xi32>)
      print(ten)

      w9 = ten[idx_ten, :, idx_ten, ...]
      # CHECK: "indexing.gather"(%[[TEN2]], %{{.*}}) {gather_dims = array<i64: 0, 1, 3, 4>} : (tensor<10x10x10x10x10xi32>, tensor<5x6x4xindex>) -> tensor<5x6x10xi32>
      print(w9.owner)

      return w1, w2, w3, w4, w5, w6, w7, w8, w9

  pm = PassManager.parse("builtin.module(func.func(cse))")
  pm.run(module.operation)
  # CHECK: module {
  # CHECK:   func.func @test_tensor_value() -> (tensor<5x6x10x10x10xi32>, tensor<5x6x10x10xi32>, tensor<5x6x10x10xi32>, tensor<5x6x10xi32>, tensor<5x6x10x10xi32>, tensor<5x6x10x10xi32>, tensor<5x6x10x10xi32>, tensor<5x6x10x10xi32>, tensor<5x6x10xi32>) {
  # CHECK:     %[[TEN:.*]] = tensor.empty() : tensor<10x10x10x10xi32>
  # CHECK:     %cst = arith.constant dense<{{.*}}> : tensor<5x6x1xindex>
  # CHECK:     %1 = indexing.gather %[[TEN]][%cst] gather_dims([0]) : (tensor<10x10x10x10xi32>, tensor<5x6x1xindex>) -> tensor<5x6x10x10x10xi32>
  # CHECK:     %cst_0 = arith.constant dense<{{.*}}> : tensor<5x6x2xindex>
  # CHECK:     %2 = indexing.gather %[[TEN]][%cst_0] gather_dims([0, 1]) : (tensor<10x10x10x10xi32>, tensor<5x6x2xindex>) -> tensor<5x6x10x10xi32>
  # CHECK:     %3 = indexing.gather %[[TEN]][%cst_0] gather_dims([0, 2]) : (tensor<10x10x10x10xi32>, tensor<5x6x2xindex>) -> tensor<5x6x10x10xi32>
  # CHECK:     %cst_1 = arith.constant dense<{{.*}}> : tensor<5x6x3xindex>
  # CHECK:     %4 = indexing.gather %[[TEN]][%cst_1] gather_dims([0, 2, 3]) : (tensor<10x10x10x10xi32>, tensor<5x6x3xindex>) -> tensor<5x6x10xi32>
  # CHECK:     %cst_2 = arith.constant dense<{{.*}}> : tensor<5x6x2xindex>
  # CHECK:     %5 = indexing.gather %[[TEN]][%cst_2] gather_dims([0, 1]) : (tensor<10x10x10x10xi32>, tensor<5x6x2xindex>) -> tensor<5x6x10x10xi32>
  # CHECK:     %6 = indexing.gather %[[TEN]][%cst_2] gather_dims([0, 2]) : (tensor<10x10x10x10xi32>, tensor<5x6x2xindex>) -> tensor<5x6x10x10xi32>
  # CHECK:     %7 = indexing.gather %[[TEN]][%cst_2] gather_dims([1, 2]) : (tensor<10x10x10x10xi32>, tensor<5x6x2xindex>) -> tensor<5x6x10x10xi32>
  # CHECK:     %8 = tensor.empty() : tensor<10x10x10x10x10xi32>
  # CHECK:     %cst_3 = arith.constant dense<{{.*}}> : tensor<5x6x4xindex>
  # CHECK:     %9 = indexing.gather %8[%cst_3] gather_dims([0, 1, 3, 4]) : (tensor<10x10x10x10x10xi32>, tensor<5x6x4xindex>) -> tensor<5x6x10xi32>
  # CHECK:     return %1, %2, %3, %4, %5, %6, %5, %7, %9 : tensor<5x6x10x10x10xi32>, tensor<5x6x10x10xi32>, tensor<5x6x10x10xi32>, tensor<5x6x10xi32>, tensor<5x6x10x10xi32>, tensor<5x6x10x10xi32>, tensor<5x6x10x10xi32>, tensor<5x6x10x10xi32>, tensor<5x6x10xi32>
  # CHECK:   }
  # CHECK: }
  print(module)


# CHECK-LABEL: TEST: testNontrivialSlicing
@run
def testNontrivialSlicing():
  i32 = IntegerType.get_signless(32)
  with mlir_mod_ctx() as module:

    @func.FuncOp.from_py_func()
    def test_tensor_value():
      ten = Tensor.empty((10, 10, 10, 10), i32)
      # CHECK: Tensor(%[[TEN:.*]], tensor<10x10x10x10xi32>)
      print(ten)

      w = ten[2:4]
      # CHECK: %{{.*}} = "indexing.gather"(%0, %1) {gather_dims = array<i64: 0>, unique} : (tensor<10x10x10x10xi32>, tensor<2x1xindex>) -> tensor<2x10x10x10xi32>
      print(w.owner)

      return w

  pm = PassManager.parse("builtin.module(func.func(cse))")
  pm.run(module.operation)
  # CHECK: module {
  # CHECK:   func.func @test_tensor_value() -> tensor<2x10x10x10xi32> {
  # CHECK:     %0 = tensor.empty() : tensor<10x10x10x10xi32>
  # CHECK:     %cst = arith.constant dense<{{\[}}[2], [3]]> : tensor<2x1xindex>
  # CHECK:     %1 = indexing.gather %0[%cst] gather_dims([0]) unique : (tensor<10x10x10x10xi32>, tensor<2x1xindex>) -> tensor<2x10x10x10xi32>
  # CHECK:     return %1 : tensor<2x10x10x10xi32>
  # CHECK:   }
  # CHECK: }
  print(module)
