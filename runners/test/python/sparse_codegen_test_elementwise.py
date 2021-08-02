"""Testing elementwise operations to explore annotation and codegen options.

This module tests the elementwise add operation with all combinations of
sparsity annotation and codegen options.
"""

from typing import List
import itertools
import sys

import numpy as np

# Import MLIR related modules.
from mlir import ir
from mlir.dialects import sparse_tensor as st
from mlir.dialects.linalg.opdsl import lang as dsl

# Import common test tools.
import sparse_codegen_test_common as test_common


# TODO(b/195340661): Parallelize the tests.
# TODO(b/195340661): Allow the tests to continue until the number of failures
# reaches a threshold.
# TODO(b/195340661): Allow inputs with different values for (pw, iw).
def _test_add(test_desc: test_common.TestDesc, a2: List[st.DimLevelType],
              b2: List[st.DimLevelType], so2: List[int], pw: int, iw: int,
              ps: int, vl: int):
  """Generates and runs C = A + B to explore annotation and codegen options.

  Args:
    test_desc: The test descriptor.
    a2: The sparsity of A and is a list of DimLevelType.
    b2: The sparsity of B and is a list of DimLevelType.
    so2: The dimension ordering for A and B. Its value is a list of integers.
    pw: The integer pointer bitwidth.
    iw: The integer index bitwidth.
    ps: The integer parallelization strategy.
    vl: The integer vector length.
  """
  test_name = f"test_{a2}_{b2}_{so2}_{pw}_{iw}_{ps}_{vl}"
  with ir.Context() as ctx:
    actual_result = test_desc.get_result(ps, vl, [
        test_common.InputDesc(so2, a2, pw, iw),
        test_common.InputDesc(so2, b2, pw, iw)
    ])

    if np.allclose(actual_result, test_desc.get_reference_result):
      print(test_name, " passed")
    else:
      print(test_name, " failed")
      sys.exit("FAILURE")


def run_test():
  # Defines the test descriptor. The operation being test will be assigned
  # later.
  test_desc = test_common.TestDesc("add", [dsl.S.M, dsl.S.N], [8, 16],
                                   [dsl.S.M, dsl.S.N], [dsl.S.M, dsl.S.N],
                                   [dsl.S.M, dsl.S.N])

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
  for a2, b2, so2, pw, iw, pa, vl in itertools.product(
      test_common.sparsities2(), test_common.sparsities2(),
      test_common.orderings2(), test_common.bitwidths(),
      test_common.bitwidths(), test_common.pars(), test_common.vls()):
    _test_add(test_desc, a2, b2, so2, pw, iw, pa, vl)


run_test()
