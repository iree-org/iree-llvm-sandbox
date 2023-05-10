# RUN: %PYTHON %s | FileCheck %s
from itertools import permutations
from random import random

import numpy as np

from mlir_structured._mlir_libs._mlir.ir import IndexType, F32Type
from mlir_structured.dialects import arith, indexing
from mlir_structured.dialects.indexing import Scalar, Tensor, IndexTensorType, _canonicalize_tuple_index
from mlir_structured.ir import Context, IntegerType, F64Type
from mlir_structured.passmanager import PassManager
from mlir_structured.runtime.util import mlir_mod_ctx


def get_array_on_one_line(a):
  return np.array_str(a, max_line_width=np.inf).replace("\n", ",")


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
  index = IndexType.get()
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

    zero_index = Scalar(0, dtype=index)
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
    ten = Tensor.empty((10, 10), i32)
    # CHECK: Tensor(%[[TEN:.*]], tensor<10x10xi32>)
    print(ten)

    concat_single_ten_first_dim = indexing.ConcatenateOp((ten,), 0).result
    # CHECK: %{{.}} = indexing.concatenate(%[[TEN]]) {dim = 0} : (tensor<10x10xi32>) -> tensor<10x10xi32>
    print(concat_single_ten_first_dim.owner)

    concat_ten_first_dim = indexing.ConcatenateOp((ten, ten), 0).result
    # CHECK: %{{.*}} = indexing.concatenate(%[[TEN]], %[[TEN]]) {dim = 0} : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<20x10xi32>
    print(concat_ten_first_dim.owner)

    concat_ten_second_dim = indexing.ConcatenateOp((ten, ten), 1).result
    # CHECK: %{{.*}} = indexing.concatenate(%[[TEN]], %[[TEN]]) {dim = 1} : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<10x20xi32>
    print(concat_ten_second_dim.owner)

    concat_ten_first_dim = indexing.concatenate((ten, ten), 0)
    # CHECK: %{{.*}} = indexing.concatenate(%[[TEN]], %[[TEN]]) {dim = 0} : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<20x10xi32>
    print(concat_ten_first_dim.owner)

    concat_ten_second_dim = indexing.concatenate((ten, ten), 1)
    # CHECK: %{{.*}} = indexing.concatenate(%[[TEN]], %[[TEN]]) {dim = 1} : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<10x20xi32>
    print(concat_ten_second_dim.owner)

    x = np.random.random((10, 10))
    ten_x = Tensor(x)
    concat_x = indexing.concatenate([ten_x, ten_x], 1)
    # CHECK: %{{.*}} = arith.constant dense<{{.*}}> : tensor<10x20xf64>
    print(concat_x.owner)
    # CHECK: True
    print(np.allclose(concat_x.literal_value, np.concatenate([x, x], axis=1)))


# CHECK-LABEL: TEST: testSimpleLiteralIndexing
@run
def testSimpleLiteralIndexing():
  i32 = IntegerType.get_signless(32)
  with mlir_mod_ctx() as module:

    ten = Tensor.empty((10, 10, 10, 10), i32)
    # CHECK: %[[TEN:.*]]
    print(ten.get_name())

    w = ten[0]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1xindex>, [0])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0]) unique : (tensor<10x10x10x10xi32>, tensor<1xindex>) -> tensor<10x10x10xi32>
    print(w.owner)

    w = ten[2, 4]
    # CHECK: Tensor(%[[CST0:.*]], tensor<2xindex>, [2 4])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0, 1]) unique : (tensor<10x10x10x10xi32>, tensor<2xindex>) -> tensor<10x10xi32>
    print(w.owner)

    w = ten[2, 4, 6]
    # CHECK: Tensor(%[[CST0:.*]], tensor<3xindex>, [2 4 6])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0, 1, 2]) unique : (tensor<10x10x10x10xi32>, tensor<3xindex>) -> tensor<10xi32>
    print(w.owner)

    w = ten[2, 4, 6, 8]
    # CHECK: Scalar(%[[CST1:.*]], index, 2)
    print(Scalar(w.owner.operands[1]))
    # CHECK: Scalar(%[[CST2:.*]], index, 4)
    print(Scalar(w.owner.operands[2]))
    # CHECK: Scalar(%[[CST3:.*]], index, 6)
    print(Scalar(w.owner.operands[3]))
    # CHECK: Scalar(%[[CST4:.*]], index, 8)
    print(Scalar(w.owner.operands[4]))
    # CHECK: %extracted = tensor.extract %[[TEN]][%[[CST1]], %[[CST2]], %[[CST3]], %[[CST4]]] : tensor<10x10x10x10xi32>
    print(w.owner)

    w = ten[...]
    # CHECK: %[[TEN]]
    print(w.get_name())

    w = ten[:]
    # CHECK: %[[TEN]]
    print(w.get_name())

    w = ten[:, :]
    # CHECK: %[[TEN]]
    print(w.get_name())

    w = ten[:, :, :]
    # CHECK: %[[TEN]]
    print(w.get_name())

    w = ten[:, :, :, :]
    # CHECK: %[[TEN]]
    print(w.get_name())

    w = ten[1, ...]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1xindex>, [1])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0]) unique : (tensor<10x10x10x10xi32>, tensor<1xindex>) -> tensor<10x10x10xi32>
    print(w.owner)

    w = ten[1, :, ...]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1xindex>, [1])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0]) unique : (tensor<10x10x10x10xi32>, tensor<1xindex>) -> tensor<10x10x10xi32>
    print(w.owner)

    w = ten[1, :, :, ...]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1xindex>, [1])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0]) unique : (tensor<10x10x10x10xi32>, tensor<1xindex>) -> tensor<10x10x10xi32>
    print(w.owner)

    try:
      w = ten[1, :, :, :, :]
    except IndexError as e:
      # CHECK: Too many indices for tensor: 5 non-None/Ellipsis indices for dim 4.
      print(e)

    w = ten[1, :]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1xindex>, [1])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0]) unique : (tensor<10x10x10x10xi32>, tensor<1xindex>) -> tensor<10x10x10xi32>
    print(w.owner)

    w = ten[1, :, :]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1xindex>, [1])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0]) unique : (tensor<10x10x10x10xi32>, tensor<1xindex>) -> tensor<10x10x10xi32>
    print(w.owner)

    w = ten[1, :, :, :]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1xindex>, [1])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0]) unique : (tensor<10x10x10x10xi32>, tensor<1xindex>) -> tensor<10x10x10xi32>
    print(w.owner)

    w = ten[:, 1]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1xindex>, [1])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([1]) unique : (tensor<10x10x10x10xi32>, tensor<1xindex>) -> tensor<10x10x10xi32>
    print(w.owner)

    w = ten[:, :, 1]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1xindex>, [1])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([2]) unique : (tensor<10x10x10x10xi32>, tensor<1xindex>) -> tensor<10x10x10xi32>
    print(w.owner)

    w = ten[:, :, :, 1]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1xindex>, [1])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([3]) unique : (tensor<10x10x10x10xi32>, tensor<1xindex>) -> tensor<10x10x10xi32>
    print(w.owner)

    w = ten[:, 1, :, 1]
    # CHECK: Tensor(%[[CST0:.*]], tensor<2xindex>, [1 1])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([1, 3]) unique : (tensor<10x10x10x10xi32>, tensor<2xindex>) -> tensor<10x10xi32>
    print(w.owner)

    w = ten[1, :, :, 1]
    # CHECK: Tensor(%[[CST0:.*]], tensor<2xindex>, [1 1])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0, 3]) unique : (tensor<10x10x10x10xi32>, tensor<2xindex>) -> tensor<10x10xi32>
    print(w.owner)

    w = ten[1, 1, :, :]
    # CHECK: Tensor(%[[CST0:.*]], tensor<2xindex>, [1 1])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0, 1]) unique : (tensor<10x10x10x10xi32>, tensor<2xindex>) -> tensor<10x10xi32>
    print(w.owner)

    w = ten[:, :, 1, 1]
    # CHECK: Tensor(%[[CST0:.*]], tensor<2xindex>, [1 1])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([2, 3]) unique : (tensor<10x10x10x10xi32>, tensor<2xindex>) -> tensor<10x10xi32>
    print(w.owner)

    w = ten[:, 1, 1, :]
    # CHECK: Tensor(%[[CST0:.*]], tensor<2xindex>, [1 1])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([1, 2]) unique : (tensor<10x10x10x10xi32>, tensor<2xindex>) -> tensor<10x10xi32>
    print(w.owner)

    w = ten[1, :, 1, :]
    # CHECK: Tensor(%[[CST0:.*]], tensor<2xindex>, [1 1])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0, 2]) unique : (tensor<10x10x10x10xi32>, tensor<2xindex>) -> tensor<10x10xi32>
    print(w.owner)

    w = ten[1, 1, :, 1]
    # CHECK: Tensor(%[[CST0:.*]], tensor<3xindex>, [1 1 1])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0, 1, 3]) unique : (tensor<10x10x10x10xi32>, tensor<3xindex>) -> tensor<10xi32>
    print(w.owner)

    w = ten[1, :, 1, 1]
    # CHECK: Tensor(%[[CST0:.*]], tensor<3xindex>, [1 1 1])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0, 2, 3]) unique : (tensor<10x10x10x10xi32>, tensor<3xindex>) -> tensor<10xi32>
    print(w.owner)

    w = ten[:, 1, 1, 1]
    # CHECK: Tensor(%[[CST0:.*]], tensor<3xindex>, [1 1 1])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([1, 2, 3]) unique : (tensor<10x10x10x10xi32>, tensor<3xindex>) -> tensor<10xi32>
    print(w.owner)

    w = ten[1, 1, 1, :]
    # CHECK: Tensor(%[[CST0:.*]], tensor<3xindex>, [1 1 1])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0, 1, 2]) unique : (tensor<10x10x10x10xi32>, tensor<3xindex>) -> tensor<10xi32>
    print(w.owner)


# CHECK-LABEL: TEST: testCanonicalizeTupleIndexCastListLiteral
@run
def testCanonicalizeTupleIndexCastListLiteral():
  with mlir_mod_ctx() as module:

    for n_tens in range(1, 4):
      uniqs = set()
      n_slices = 4 - n_tens
      ten_idx = [[0], [1]]
      slice_idx = slice(None)
      for p in permutations([str(ten_idx)] * n_tens +
                            [str(slice_idx)] * n_slices):
        uniqs.add(p)

      for u in uniqs:
        u = tuple(u)
        tens_is = [i for i, t in enumerate(u) if t == str(ten_idx)]
        slice_is = [i for i, s in enumerate(u) if s == str(slice_idx)]

        tens_slices = _canonicalize_tuple_index(tuple(map(eval, u)), 4)
        tens = [
            (i, t) for i, t in enumerate(tens_slices) if isinstance(t, Tensor)
        ]
        slices = [(i, s) for i, s in enumerate(tens_slices) if s == slice(None)]
        assert len(slices) == n_slices and all(
            s == slice(None) for _, s in slices) and set(
                i for i, _ in slices) == set(slice_is)
        assert len(tens) == n_tens and all(
            isinstance(t, Tensor) and t.owner.name == 'arith.constant' and
            str(t.type) == 'tensor<2x1xindex>' and t.is_constant() and
            np.array_equal(t.literal_value, [[0], [1]])
            for _, t in tens) and set(i for i, _ in tens) == set(tens_is)


# CHECK-LABEL: TEST: testAdvancedIndexing
@run
def testAdvancedIndexing():
  index = IndexType.get()
  f32 = F32Type.get()
  with mlir_mod_ctx() as module:
    ten = Tensor.empty((10, 10, 10, 10), f32)
    # CHECK: Tensor(%[[TEN:.*]], tensor<10x10x10x10xf32>)
    print(ten)

    w = ten[[[0], [1]], :, :, :]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<2x1xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK{LITERAL}: [[0], [1]]
    print(get_array_on_one_line(idx_tensor_operand.literal_value))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0]) unique : (tensor<10x10x10x10xf32>, tensor<2x1xindex>) -> tensor<2x10x10x10xf32>
    print(w.owner)

    w = ten[[[0], [1]], [[0], [1]], :, :]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<2x2xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK{LITERAL}: [[0 0], [1 1]]
    print(get_array_on_one_line(idx_tensor_operand.literal_value))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0, 1]) unique : (tensor<10x10x10x10xf32>, tensor<2x2xindex>) -> tensor<2x10x10xf32>
    print(w.owner)

    w = ten[[[0], [1]], :, [[0], [1]], :]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<2x2xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK{LITERAL}: [[0 0], [1 1]]
    print(get_array_on_one_line(idx_tensor_operand.literal_value))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0, 2]) unique : (tensor<10x10x10x10xf32>, tensor<2x2xindex>) -> tensor<2x10x10xf32>
    print(w.owner)

    # ensure that the indices aren't unique because the checks below all except non-unique
    idx_tensor = np.random.randint(low=0, high=10, size=(5, 6, 1))
    while len(np.unique(idx_tensor)) == idx_tensor.size:
      idx_tensor = np.random.randint(low=0, high=10, size=(5, 6, 1))
    idx_tensor = Tensor(idx_tensor, dtype=index)
    # CHECK: %[[IDXTEN:.*]] tensor<5x6x1xindex> True
    print(idx_tensor.get_name(), idx_tensor.type, idx_tensor.is_constant())
    # CHECK: %[[IDXTEN:.*]] = arith.constant dense<{{.*}}> : tensor<5x6x1xindex>
    print(idx_tensor.owner)
    # CHECK: True
    print(np.array(idx_tensor.literal_value).dtype == np.int64)

    w = ten[idx_tensor, ...]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<5x6x1xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0]) : (tensor<10x10x10x10xf32>, tensor<5x6x1xindex>) -> tensor<5x6x10x10x10xf32>
    print(w.owner)

    w = ten[idx_tensor, idx_tensor, ...]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<5x6x2xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0, 1]) : (tensor<10x10x10x10xf32>, tensor<5x6x2xindex>) -> tensor<5x6x10x10xf32>
    print(w.owner)

    w = ten[idx_tensor, :, idx_tensor]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<5x6x2xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0, 2]) : (tensor<10x10x10x10xf32>, tensor<5x6x2xindex>) -> tensor<5x6x10x10xf32>
    print(w.owner)

    w = ten[idx_tensor, :, idx_tensor, idx_tensor]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<5x6x3xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0, 2, 3]) : (tensor<10x10x10x10xf32>, tensor<5x6x3xindex>) -> tensor<5x6x10xf32>
    print(w.owner)

    idx_tensor = np.random.randint(low=0, high=10, size=(5, 6, 2))
    while len(np.unique(idx_tensor)) == idx_tensor.size:
      idx_tensor = np.random.randint(low=0, high=10, size=(5, 6, 2))
    idx_tensor = Tensor(idx_tensor, dtype=index)
    # CHECK: %[[IDXTEN:.*]] = arith.constant dense<{{.*}}> : tensor<5x6x2xindex>
    print(idx_tensor.owner)

    w = indexing.gather(ten, idx_tensor, [0, 1])
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN]] tensor<5x6x2xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0, 1]) : (tensor<10x10x10x10xf32>, tensor<5x6x2xindex>) -> tensor<5x6x10x10xf32>
    print(w.owner)

    w = indexing.gather(ten, idx_tensor, [0, 2])
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN]] tensor<5x6x2xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0, 2]) : (tensor<10x10x10x10xf32>, tensor<5x6x2xindex>) -> tensor<5x6x10x10xf32>
    print(w.owner)

    w = ten[idx_tensor, ...]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<5x6x2xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0, 1]) : (tensor<10x10x10x10xf32>, tensor<5x6x2xindex>) -> tensor<5x6x10x10xf32>
    print(w.owner)

    w = ten[:, idx_tensor, ...]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<5x6x2xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([1, 2]) : (tensor<10x10x10x10xf32>, tensor<5x6x2xindex>) -> tensor<5x6x10x10xf32>
    print(w.owner)

    ten = Tensor.empty((10, 10, 10, 10, 10), f32)
    # CHECK: Tensor(%[[TEN:.*]], tensor<10x10x10x10x10xf32>)
    print(ten)

    w = ten[idx_tensor, :, idx_tensor, ...]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<5x6x4xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0, 1, 3, 4]) : (tensor<10x10x10x10x10xf32>, tensor<5x6x4xindex>) -> tensor<5x6x10xf32>
    print(w.owner)

    w = ten[idx_tensor, 0:10:1, idx_tensor, ...]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<5x6x4xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0, 1, 3, 4]) : (tensor<10x10x10x10x10xf32>, tensor<5x6x4xindex>) -> tensor<5x6x10xf32>
    print(w.owner)

    try:
      w = ten[idx_tensor, 0:10:2, idx_tensor, ...]
    except IndexError as e:
      # CHECK: Partial slicing currently not supported
      print(e)
