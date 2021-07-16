"""Testing elementwise operations to explore annotation and codegen options.

This module tests the elementwise add operation with all combinations of
sparsity annotation and codegen options.
"""

import itertools
from typing import List

import numpy as np

# Import MLIR related modules.
from mlir import ir
from mlir.dialects import sparse_tensor as st
from mlir.dialects.linalg.opdsl import lang as dsl

# Import common test tools.
import sparse_codegen_test_common as tc


# TODO:
# Uses generated test data and more general tensor dimensions.
# Parallelizes the tests.
# Allows the tests to continue until the number of failures reaches a threshold.
# Allows inputs with different values for (pw, iw).
def _test_add(test_desc: tc.TestDesc, a2: List[st.DimLevelType],
              b2: List[st.DimLevelType], so2: List[int], pw: int, iw: int,
              pa: int, vl: int):
  """Generates and runs C = A + B to explore annotation and codegen options.

  Args:
    test_desc: The test descriptor.
    a2: The sparsity of A and is a list of DimLevelType.
    b2: The sparsity of B and is a list of DimLevelType.
    so2: The dimension ordering for A and B. Its value is a list of integers.
    pw: The integer pointer bitwidth.
    iw: The integer index bitwidth.
    pa: The integer parallelization strategy.
    vl: The integer vector length.
  """
  test_name = f"test_{a2}_{b2}_{so2}_{pw}_{iw}_{pa}_{vl}"
  with ir.Context() as ctx:
    actual_result = test_desc.get_result(
        pa, vl, [tc.InputDesc(so2, a2, pw, iw),
                 tc.InputDesc(so2, b2, pw, iw)])

    if np.allclose(actual_result, test_desc.reference_result):
      print(test_name, " passed")
    else:
      print(test_name, " failed")
      quit("FAILURE")


def run_test():
  # Defines the test descriptor. The operation being test will be assigned
  # later.
  test_desc = tc.TestDesc([dsl.S.M, dsl.S.N], [8, 8], [dsl.S.M, dsl.S.N],
                          [dsl.S.M, dsl.S.N], [dsl.S.M, dsl.S.N])

  @dsl.linalg_structured_op
  def add_dsl(
      A=dsl.TensorDef(dsl.T, *test_desc.inputs[0]),
      B=dsl.TensorDef(dsl.T, *test_desc.inputs[1]),
      C=dsl.TensorDef(dsl.T, *test_desc.output, output=True)):
    """The operation being tested: C = A + B."""
    C[dsl.D.m, dsl.D.n] = A[dsl.D.m, dsl.D.n] + B[dsl.D.m, dsl.D.n]

  # Pass the operation to the test descriptor.
  test_desc.linalg_op = add_dsl

  # Calculate the reference result.
  test_desc.calculate_reference_result()

  # Test all combinations of annotations and codegen options.
  for a2, b2, so2, pw, iw, pa, vl in itertools.product(tc.sparsities2(),
                                                       tc.sparsities2(),
                                                       tc.orderings2(),
                                                       tc.bitwidths(),
                                                       tc.bitwidths(),
                                                       tc.pars(), tc.vls()):
    _test_add(test_desc, a2, b2, so2, pw, iw, pa, vl)


run_test()
