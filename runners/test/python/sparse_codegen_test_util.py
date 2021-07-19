"""Testing error handling in the common test utilities.

The module tests the error handling in the utilities we use for writing
exhaustive tests for the sparse codegen.
"""

import inspect
import sys

# Import MLIR related modules.
from mlir.dialects import sparse_tensor as st
from mlir.dialects.linalg.opdsl import lang as dsl

# Import common test tools.
import sparse_codegen_test_common as tc

# A test returns 1 when it fails to indicate the number of failing test. This is
# to help accumulating the total number of failing tests.
_PASS = 0
_FAIL = 1


def _pass_test(name: str) -> int:
  print(f"{name} passed.")
  return _PASS


def _fail_test(name: str) -> int:
  print(f"{name} failed.")
  return _FAIL


def _test_mismatching_ordering_sparsity() -> int:
  """Test for inconsistent input descriptor parameters.

  The dimension ordering and the sparsities in this test don't have the same
  length.
  """
  name = inspect.currentframe().f_code.co_name
  try:
    _ = tc.InputDesc([0, 1, 2], [st.DimLevelType.dense, st.DimLevelType.dense],
                     0, 0)
  except ValueError:
    num_failed = _pass_test(name)
  else:
    num_failed = _fail_test(name)
  return num_failed


def _test_invalid_ordering() -> int:
  """Test for invalid dimension orderings.

  The dimension ordering in this test is not a permutation of 0..n-1, where
  n is the length of the dimension ordering.
  """
  name = inspect.currentframe().f_code.co_name
  try:
    _ = tc.InputDesc([0, 2], [st.DimLevelType.dense, st.DimLevelType.dense], 0,
                     0)
  except ValueError:
    num_failed = _pass_test(name)
  else:
    num_failed = _fail_test(name)
  return num_failed


def _test_invalid_affine_expression() -> int:
  """Test for invalid affine expressions.

  The affine expression in the first input here is not defined in the iteration
  space.
  """
  name = inspect.currentframe().f_code.co_name
  try:
    _ = tc.TestDesc([dsl.S.M, dsl.S.N], [8, 8], [dsl.S.M, dsl.S.X])
  except ValueError:
    num_failed = _pass_test(name)
  else:
    num_failed = _fail_test(name)
  return num_failed


def run_test():
  num_failed = (
      _test_mismatching_ordering_sparsity() + _test_invalid_ordering() +
      _test_invalid_affine_expression())

  if num_failed == 0:
    print("All test passed.")
  else:
    print(f"{num_failed} tests failed.")
    sys.exit("FAILURE")


run_test()
