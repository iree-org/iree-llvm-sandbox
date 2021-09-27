"""Testing error handling in the common test utilities.

This module unit-tests the utilities we use for writing exhaustive test tools
for the MLIR sparse codegen.
"""

from typing import Callable, List
import inspect
import itertools
import logging
import numpy as np
import sys

# Import MLIR related modules.
from mlir.dialects import sparse_tensor as st
from mlir.dialects.linalg.opdsl import lang as dsl

# Import common test tools.
from . import sparse_codegen_test_common as test_common


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


@_test_unit_wrapper
def _test_float_sparse_tensor_types() -> bool:
  """Tests the supported floating point sparse tensor type combinations.

  For floating point types, all combinations of (type, pointer-bitwidth,
  index-bitwidth) are supported.
  """
  for comb in itertools.product(test_common.fp_types(), test_common.bitwidths(),
                                test_common.bitwidths()):
    if not test_common.supported_tensor_types(*comb):
      return False

  return True


@_test_unit_wrapper
def _test_int_sparse_tensor_types() -> bool:
  """Tests the supported integer sparse tensor type combinations.

  Checks the supported combination of (type, pointer-bitwidth, index-bitwidth)
  where type is an integer type.

  For integer types, we only support the cases where pointer-bitwidth and
  index-bitwidth have the same value. For I64, we only support 64 bitwidth.
  """
  for comb in itertools.product(test_common.int_types(),
                                test_common.bitwidths(),
                                test_common.bitwidths()):
    supported = (
        comb[1] == comb[2] and
        (comb[0] != test_common.TDType.I64 or comb[1] == 64))

    if supported != test_common.supported_tensor_types(*comb):
      return False

  return True


@_test_unit_wrapper
def _test_generate_tensor() -> bool:
  """Tests the tensor generating function.

  In particular, checks that function returns a list of integer with the same
  number of elements as the given shape.
  """
  shape = [2, 3, 4]
  array = test_common.generate_tensor(shape)
  return len(array) == np.prod(shape) and isinstance(array[0], int)


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
    reference_result = test_desc.get_reference_result(test_common.TDType.F64)
  except ValueError:
    return True
  else:
    return False


@_test_unit_wrapper
def _test_reference_result() -> bool:
  """Verifies the shape of the reference result returned by a TestDesc object.

  This is to make sure that the program runs and the reference result is a numpy
  array with the expected shape and data type.
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
  test_desc.calculate_reference_result(test_common.TDType.F64)

  result_type = test_common.TDType.I32
  array = test_desc.get_reference_result(result_type)

  return isinstance(array, np.ndarray) and tuple(
      shape) == array.shape and array.dtype == result_type.value


def run_test() -> None:
  logging.basicConfig(level=logging.INFO)
  num_failed = (
      _test_mismatching_ordering_sparsity() + _test_invalid_ordering() +
      _test_invalid_affine_expression() + _test_generate_tensor() +
      _test_float_sparse_tensor_types() + _test_int_sparse_tensor_types() +
      _test_reference_result_before_available() + _test_reference_result())

  if num_failed:
    logging.error(f"{num_failed} tests failed.")
    sys.exit(test_common.FAILURE_MESSAGE)
  else:
    logging.info("All test passed.")


def main():
  run_test()


if __name__ == "__main__":
  main()
