from mlir.ir import *
from mlir.dialects.linalg.opdsl.lang import *

################################################################################
### Copy
################################################################################


# TODO: fold OpDSL definition and inferences into ProblemDefinition.
@linalg_structured_op
def copy_1d(I=TensorDef(T, S.M), O=TensorDef(T, S.M, output=True)):
  domain(D.m)
  O[D.m] = I[D.m]


@linalg_structured_op
def copy_2d(I=TensorDef(T, S.N, S.M), O=TensorDef(T, S.M, S.N, output=True)):
  domain(D.m, D.n)
  O[D.m, D.n] = I[D.m, D.n]


@linalg_structured_op
def copy_3d(I=TensorDef(T, S.M, S.N, S.K),
            O=TensorDef(T, S.M, S.N, S.K, output=True)):
  domain(D.m, D.n, D.k)
  O[D.m, D.n, D.k] = I[D.m, D.n, D.k]


@linalg_structured_op
def copy_4d(I=TensorDef(T, S.M, S.N, S.K, S.L),
            O=TensorDef(T, S.M, S.N, S.K, S.L, output=True)):
  domain(D.m, D.n, D.k, D.l)
  O[D.m, D.n, D.k, D.l] = I[D.m, D.n, D.k, D.l]
