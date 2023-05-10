# RUN: %PYTHON %s | FileCheck %s
from random import random

import numpy as np

from mlir_structured.dialects import func, arith, indexing
from mlir_structured.dialects.indexing import Scalar, Tensor, IndexTensorType
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

    no_fold_one_f64 = Scalar(1.0, fold=False)
    # CHECK: Scalar(%[[NF1:.*]], f64, 1.0)
    print(no_fold_one_f64)
    no_fold_two_f64 = Scalar(2.0, fold=False)
    # CHECK: Scalar(%[[NF2:.*]], f64, 2.0)
    print(no_fold_two_f64)

    no_fold_three_f64 = no_fold_one_f64 + no_fold_two_f64
    # CHECK: %{{.*}} = arith.addf %[[NF1]], %[[NF2]] : f64
    print(no_fold_three_f64.owner)
    # CHECK: False
    print(no_fold_three_f64.is_constant())


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
