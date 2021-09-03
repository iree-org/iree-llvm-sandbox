"""Testing MatMul operations to explore annotation and codegen options.

This module tests the matrix multiplication operation with all combinations of
sparsity annotation and codegen options. If there is any set of parameters that
triggers a test failure, the module raises a SystemExit exception.

TODO(b/195340661): Allow the tests to continue until the number of failures
reaches a threshold specified by the user.
"""

from typing import List, Tuple
import itertools
import numpy as np
import sys

# Import MLIR related modules.
from mlir import ir
from mlir.dialects import sparse_tensor as st
from mlir.dialects.linalg.opdsl import lang as dsl

# Import common test tools.
import sparse_codegen_test_common as test_common


# TODO(b/195422626): Make _test_desc a local variable and pass it to
# routine _test_print after fixing the issue.
#
# Defines the test descriptor. The operation being test will be assigned
# later. Notation dsl.S.X defines a symbol in the domain specific language
# dialect of mlir.dialects.linalg.opdsl as a tensor dimension, where X is the
# name of the symbol.
_test_desc = test_common.TestDesc("matmul", [dsl.S.M, dsl.S.N, dsl.S.K],
                                  [8, 8, 8], [dsl.S.M, dsl.S.N],
                                  [dsl.S.M, dsl.S.K], [dsl.S.K, dsl.S.N])


def _test_combination_impl(a2: List[st.DimLevelType], b2: List[st.DimLevelType],
                           a_so2: List[int], b_so2: List[int], a_pw: int,
                           b_pw: int, a_iw: int, b_iw: int, ps: int,
                           vl: int) -> bool:
  """Runs the test for the given parameter combination.

  Args:
    test_desc: The test descriptor.
    a2: The sparsity of A and is a list of DimLevelType.
    b2: The sparsity of B and is a list of DimLevelType.
    a_so2: The dimension ordering for A. Its value is a list of integers.
    b_so2: The dimension ordering for B. Its value is a list of integers.
    a_pw: The integer pointer bitwidth for A.
    b_pw: The integer pointer bitwidth for B.
    a_iw: The integer index bitwidth for A.
    b_iw: The integer index bitwidth for B.
    ps: The integer parallelization strategy.
    vl: The integer vector length.

  Returns:
    A boolean to indicate whether the test passes (True) or fails (False).
  """
  with ir.Context() as ctx:
    actual_result = _test_desc.get_result(ps, vl, [
        test_common.InputDesc(a_so2, a2, a_pw, a_iw),
        test_common.InputDesc(b_so2, b2, b_pw, b_iw)
    ])

    passed = np.allclose(actual_result, _test_desc.reference_result)

  return passed


# Python multiprocessing doesn't support decorators. We have to use a wrapped
# function in the module scope instead.
#
# See https://stackoverflow.com/questions/9336646/python-decorator-with-multiprocessing-fails
def _test_combination(*args):
  return test_common.test_combination_wrapper(_test_combination_impl)(*args)


def _parameter_combinations(
) -> Tuple[st.DimLevelType, st.DimLevelType, List[int], List[int], int, int,
           int, int, int, int]:
  """Returns all parameter combinations for the matrix multiplication test.

    The matrix multiplication test currently tests rank 2 input tensors. As
    such, the cardinalities for sparsities, ordering, bitwidths, parallelization
    options (pars) and vector length (vls) are 4, 2, 4, 5, and 3 respectively.
  """
  return itertools.product(test_common.sparsities2(), test_common.sparsities2(),
                           test_common.orderings2(), test_common.orderings2(),
                           test_common.bitwidths(), test_common.bitwidths(),
                           test_common.bitwidths(), test_common.bitwidths(),
                           test_common.pars(), test_common.vls())


def run_tests(num_processes: int) -> bool:
  """Tests operation C = A * B for various codegen options and sparsities.

  Args:
    num_processes: An integer for the number of processes used to run the tests.

  Returns:
    A boolean to indicate whether all tests pass (True) or there are failing
      tests (False).
  """
  # The operation used to test the JIT compiler and runtime.
  @dsl.linalg_structured_op
  def matmul_dsl(
      A=dsl.TensorDef(dsl.T, *_test_desc.inputs[0]),
      B=dsl.TensorDef(dsl.T, *_test_desc.inputs[1]),
      C=dsl.TensorDef(dsl.T, *_test_desc.output, output=True)) -> None:
    """The operation being tested: C = A * B."""
    C[dsl.D.m, dsl.D.n] += A[dsl.D.m, dsl.D.k] * B[dsl.D.k, dsl.D.n]

  # Pass the operation to the test descriptor.
  _test_desc.linalg_op = matmul_dsl

  # Calculate the reference result.
  _test_desc.calculate_reference_result()

  return test_common.run_tests_sequential_or_parallel(num_processes,
                                                      _parameter_combinations,
                                                      _test_combination)


if test_common.get_num_processes_and_run_tests(__name__, run_tests) == False:
  sys.exit(test_common.FAILURE_MESSAGE)
