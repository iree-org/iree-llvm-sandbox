# RUN: %PYTHON %s 2>&1 | FileCheck %s

# Check that various einsum specifications can actually be compiled. This is not
# a performance benchmark so we are not doing transformations other than
# lowering.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import *


def main():
  test_harness(lambda sizes, types: EinsumProblem("klnp,nk->pl"),
               [[np.float32] * 3], [{
                   "k": 10,
                   "l": 12,
                   "n": 14,
                   "p": 16
               }], {'Expert': LoweringOnlyExpert('einsum', 'linalg.generic')},
               n_iters=1,
               function_name='einsum')


if __name__ == "__main__":
  main()
