"""Testing error handling in the common test utilities.

This module unit-tests the utilities we use for writing exhaustive test tools
for the MLIR sparse codegen.
"""

from typing import Callable, List
import inspect
import logging
import numpy as np
import sys

# Import MLIR related modules.
from mlir.dialects import sparse_tensor as st
from mlir.dialects.linalg.opdsl import lang as dsl

# Import common test tools.
import sparse_codegen_test_common as test_common


def _test_unit_wrapper(test_func: Callable[[], bool]) -> Callable[[], int]:
  """Wraps a test function with post processing functionality.

  In particular, the test_func wrapper invokes test_func, logs the test and its
  status, and returns a value to indicate the number of failing tests.

  Args:
    test_func: A Callable object for invoking the test. It returns a boolean
      boolean to indicate the test status of passing (True) or failing (False).

  Returns:
    A wrapper of the test_func.
  """

  def wrapper() -> int:
    passed = test_func()

    status_str = "passed" if passed else "failed"
    logging.info(f"{test_func.__name__} {status_str}.")

    return int(passed == False)

  return wrapper


@_test_unit_wrapper
def _test_mismatching_ordering_sparsity() -> bool:
  """Test for inconsistent input descriptor parameters.

  The dimension ordering and the sparsities in this test don't have the same
  length.
  """
  try:
    _ = test_common.InputDesc([0, 1, 2],
                              [st.DimLevelType.dense, st.DimLevelType.dense], 0,
                              0)
  except ValueError:
    return True
  else:
    return False


@_test_unit_wrapper
def _test_invalid_ordering() -> bool:
  """Test for invalid dimension orderings.

  The dimension ordering in this test is not a permutation of 0..n-1, where
  n is the length of the dimension ordering.
  """
  try:
    _ = test_common.InputDesc([0, 2],
                              [st.DimLevelType.dense, st.DimLevelType.dense], 0,
                              0)
  except ValueError:
    return True
  else:
    return False


@_test_unit_wrapper
def _test_invalid_affine_expression() -> bool:
  """Test for invalid affine expressions.

  The affine expression in the first input here is not defined in the iteration
  space.
  """
  try:
    _ = test_common.TestDesc("dummy", [dsl.S.M, dsl.S.N], [8, 8],
                             [dsl.S.M, dsl.S.X])
  except ValueError:
    return True
  else:
    return False


def _is_numpy_array_with_shape(shape: List[int], array: np.ndarray) -> bool:
  """Returns True iff array is a numpy array with the given shape."""
  return isinstance(array, np.ndarray) and tuple(shape) == array.shape


@_test_unit_wrapper
def _test_generate_tensor() -> bool:
  """Tests the tensor generating function.

  In particular, checks that function returns a numpy array with the expected
  shape.
  """
  shape = [2, 3, 4]
  return _is_numpy_array_with_shape(shape, test_common.generate_tensor(shape))


@_test_unit_wrapper
def _test_reference_result_before_available() -> bool:
  """Verifies that getting reference_result before it is ready is an error."""

  shape = [2, 3]
  # Defines the test descriptor. The operation being test will be assigned
  # later. Notation dsl.S.X defines a symbol in the domain specific language
  # dialect of mlir.dialects.linalg.opdsl as a tensor dimension, where X is the
  # name of the symbol.
  test_desc = test_common.TestDesc("copy", [dsl.S.M, dsl.S.N], shape,
                                   [dsl.S.M, dsl.S.N], [dsl.S.M, dsl.S.N])

  try:
    reference_result = test_desc.reference_result
  except ValueError:
    return True
  else:
    return False


@_test_unit_wrapper
def _test_reference_result() -> bool:
  """Verifies the shape of the reference result returned by a TestDesc object.

  This is to make sure that the program runs and the reference result is a numpy
  array with the expected shape.
  """
  shape = [2, 3]
  # Defines the test descriptor. The operation being test will be assigned
  # later. Notation dsl.S.X defines a symbol in the domain specific language
  # dialect of mlir.dialects.linalg.opdsl as a tensor dimension, where X is the
  # name of the symbol.
  test_desc = test_common.TestDesc("copy", [dsl.S.M, dsl.S.N], shape,
                                   [dsl.S.M, dsl.S.N], [dsl.S.M, dsl.S.N])

  @dsl.linalg_structured_op
  def copy_dsl(
      A=dsl.TensorDef(dsl.T, *test_desc.inputs[0]),
      B=dsl.TensorDef(dsl.T, *test_desc.output, output=True)) -> None:
    """The operation being tested: B = A."""
    B[dsl.D.m, dsl.D.n] = A[dsl.D.m, dsl.D.n]

  # Pass the operation to the test descriptor.
  test_desc.linalg_op = copy_dsl

  # Calculate the reference result.
  test_desc.calculate_reference_result()

  return _is_numpy_array_with_shape(shape, test_desc.reference_result)


def run_test() -> None:
  logging.basicConfig(level=logging.INFO)
  num_failed = (
      _test_mismatching_ordering_sparsity() + _test_invalid_ordering() +
      _test_invalid_affine_expression() + _test_generate_tensor() +
      _test_reference_result_before_available() + _test_reference_result())

  if num_failed:
    logging.error(f"{num_failed} tests failed.")
    sys.exit(test_common.FAILURE_MESSAGE)
  else:
    logging.info("All test passed.")


run_test()
