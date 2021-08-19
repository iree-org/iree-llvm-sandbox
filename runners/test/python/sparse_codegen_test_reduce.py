"""Testing reduce operations to explore annotation and codegen options.

This module tests the reduce operation with all combinations of sparsity
annotation and codegen options. If there is any set of parameters that triggers
a test failure, the module raises a SystemExit exception.

TODO(b/195340661): Allow the tests to continue until the number of failures
reaches a threshold specified by the user.
"""

from typing import List, Tuple
import itertools
import sys

import numpy as np

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
_test_desc = test_common.TestDesc("reduce", [dsl.S.M, dsl.S.N, dsl.S.O],
                                  [8, 4, 16], [dsl.S.M],
                                  [dsl.S.M, dsl.S.N, dsl.S.O])


def _test_print(a3: List[st.DimLevelType], a_so3: List[int], pw: int, iw: int,
                ps: int, vl: int) -> bool:
  """Generates and runs B = reduce(A) to explore annotation and codegen options.

  Also prints the test name and the passing or failing status.

  Args:
    a3: The sparsity of A and is a list of DimLevelType.
    a_so3: The dimension ordering for A. Its value is a list of integers.
    pw: The integer pointer bitwidth.
    iw: The integer index bitwidth.
    ps: The integer parallelization strategy.
    vl: The integer vector length.

  Returns:
    A boolean to indicate whether the test passes (True) or fails (False).
  """
  with ir.Context() as ctx:
    actual_result = _test_desc.get_result(
        ps, vl, [test_common.InputDesc(a_so3, a3, pw, iw)])

    status = np.allclose(actual_result, _test_desc.get_reference_result)

  test_name = f"test_{a3}_{a_so3}_{pw}_{iw}_{ps}_{vl}"
  print(test_name, " passed" if status else " failed")
  return status


def _combinations():
  """Returns all parameter combinations for the reduction test.

    The reduction test currently tests rank 3 input tensors. As such, the
    cardinalities for sparsities, ordering, bitwidths, parallelization
    options (pars) and vector length (vls) are 8, 6, 4, 5, and 3 respectively.
  """
  return itertools.product(test_common.sparsities3(), test_common.orderings3(),
                           test_common.bitwidths(), test_common.bitwidths(),
                           test_common.pars(), test_common.vls())


def run_tests(num_processes: int) -> bool:
  """Runs the tests with the given number of processes.

  Args:
    num_processes: An integer for the number of processes used to run the tests.

  Returns:
    A boolean to indicate whether all tests pass (True) or there are failing
      tests (False).
  """
  # The operation used to test the JIT compiler and runtime.
  @dsl.linalg_structured_op
  def _reduce_dsl(
      A=dsl.TensorDef(dsl.T, *_test_desc.inputs[0]),
      B=dsl.TensorDef(dsl.T, *_test_desc.output, output=True)):
    """The operation being tested: B = sum(A)."""
    B[dsl.D.m] += A[dsl.D.m, dsl.D.n, dsl.D.o]

  # Pass the operation to the test descriptor.
  _test_desc.linalg_op = _reduce_dsl

  # Calculate the reference result.
  _test_desc.calculate_reference_result()

  return test_common.run_tests_sequential_or_parallel(num_processes,
                                                      _combinations,
                                                      _test_print)


if test_common.get_num_processes_and_run_tests(__name__, run_tests) == False:
  sys.exit("FAILURE")
