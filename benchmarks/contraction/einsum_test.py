# RUN: %PYTHON %s 2>&1 | FileCheck %s

# Check that various einsum specifications can actually be compiled. This is not
# a performance benchmark so we are not doing transformations other than
# lowering.

from mlir_structured.sandbox.experts import *
from mlir_structured.sandbox.harness import *
from mlir_structured.sandbox.transforms import *

from definitions import *


def main():
  # Test two-operand problem.
  test_harness(lambda sizes, types: EinsumProblem('klnp,nk->pl', 'klnp', 2),
               [[np.float32] * 3], [{
                   "k": 10,
                   "l": 12,
                   "n": 14,
                   "p": 16
               }], [LoweringOnlyExpert('einsum', 'linalg.generic')],
               n_iters=1,
               function_name='einsum')

  # Test one-operand problem with scalar output.
  test_harness(lambda sizes, types: EinsumProblem('nk->', 'nk', 1),
               [[np.float32] * 3], [{
                   "k": 10,
                   "n": 14,
               }], [LoweringOnlyExpert('einsum', 'linalg.generic')],
               n_iters=1,
               function_name='einsum')


if __name__ == "__main__":
  main()
