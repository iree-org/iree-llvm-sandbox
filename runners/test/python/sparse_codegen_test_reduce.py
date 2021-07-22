"""Testing reduce operations to explore annotation and codegen options.

This module tests the reduce operation with all combinations of sparsity
annotation and codegen options.
"""

import itertools
import sys
from typing import List

import numpy as np

# Import MLIR related modules.
from mlir import ir
from mlir.dialects import sparse_tensor as st
from mlir.dialects.linalg.opdsl import lang as dsl

# Import common test tools.
import sparse_codegen_test_common as test_common


# TODO:
# Uses generated test data and more general tensor dimensions.
# Parallelizes the tests.
# Allows the tests to continue until the number of failures reaches a threshold.
def _test_reduce(test_desc: test_common.TestDesc, a2: List[st.DimLevelType],
                 a_so2: List[int], pw: int, iw: int, ps: int, vl: int) -> None:
  """Generates and runs B = reduce(A) to explore annotation and codegen options.

  Args:
  test_desc: The test descriptor.
    a2: The sparsity of A and is a list of DimLevelType.
    a_so2: The dimension ordering for A. Its value is a list of integers.
    pw: The integer pointer bitwidth.
    iw: The integer index bitwidth.
    ps: The integer parallelization strategy.
    vl: The integer vector length.
  """
  test_name = f"test_{a2}_{a_so2}_{pw}_{iw}_{ps}_{vl}"
  with ir.Context() as ctx:
    actual_result = test_desc.get_result(
        ps, vl, [test_common.InputDesc(a_so2, a2, pw, iw)])

    if np.allclose(actual_result, test_desc.reference_result):
      print(test_name, " passed")
    else:
      print(test_name, " failed")
      sys.exit("FAILURE")


def run_test():
  # Defines the test descriptor. The operation being test will be assigned
  # later.
  test_desc = test_common.TestDesc([dsl.S.M, dsl.S.N], [8, 8], [dsl.S.M],
                                   [dsl.S.M, dsl.S.N])

  @dsl.linalg_structured_op
  def reduce_dsl(
      A=dsl.TensorDef(dsl.T, *test_desc.inputs[0]),
      B=dsl.TensorDef(dsl.T, *test_desc.output, output=True)):
    """The operation being tested: B = sum(A)."""
    B[dsl.D.m] += A[dsl.D.m, dsl.D.n]

  # Pass the operation to the test descriptor.
  test_desc.linalg_op = reduce_dsl

  # Calculate the reference result.
  test_desc.calculate_reference_result()

  # Test all combinations of annotations and codegen options.
  for a2, a_so2, pw, iw, ps, vl in itertools.product(test_common.sparsities2(),
                                                     test_common.orderings2(),
                                                     test_common.bitwidths(),
                                                     test_common.bitwidths(),
                                                     test_common.pars(),
                                                     test_common.vls()):
    _test_reduce(test_desc, a2, a_so2, pw, iw, ps, vl)


run_test()
