# RUN: %PYTHON %s | FileCheck %s

from mlir_structured.dialects import indexing as idx
from mlir_structured.ir import Context, IntegerType

from mlir_structured.runtime.util import mlir_mod_ctx


def run(f):
  print("\nTEST:", f.__name__)
  with Context():
    idx.register_dialect()
    f()
  return f


# CHECK-LABEL: TEST: testCustomType
@run
def testCustomType():
  c = idx.CustomType.get("bob")
  # CHECK: !indexing.custom<"bob">
  print(c)


# CHECK-LABEL: TEST: testTensorValue
@run
def testTensorValue():
  i32 = IntegerType.get_signless(32)
  with mlir_mod_ctx():
    ten = idx.Tensor.empty([10, 10], i32)
    # CHECK: Tensor(%[[TEN:.*]] = tensor.empty() : tensor<10x10xi32>)
    print(ten)

    v = ten[0, 0]
    # CHECK: %{{.*}} = tensor.extract %[[TEN]][%{{.*}}, %{{.*}}] : tensor<10x10xi32>
    print(v)


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
