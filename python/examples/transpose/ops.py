from mlir.ir import *
from mlir.dialects.linalg.opdsl.lang import *

################################################################################
### Transpose
################################################################################
#   Op def: (     m,     n )
#    Iters: ({Par(), Par()})
#             I       O
#   Layout: {{n, m}, {m, n}}


# TODO: fold OpDSL definition and inferences into ProblemDefinition.
@linalg_structured_op
def transpose_2d(I=TensorDef(T, S.N, S.M),
                 O=TensorDef(T, S.M, S.N, output=True)):
  domain(D.m, D.n)
  O[D.m, D.n] = I[D.n, D.m]


@linalg_structured_op
def transpose_3d_012(I=TensorDef(T, S.M, S.N, S.K),
                     O=TensorDef(T, S.M, S.N, S.K, output=True)):
  domain(D.m, D.n, D.k)
  O[D.m, D.n, D.k] = I[D.m, D.n, D.k]


@linalg_structured_op
def transpose_3d_021(I=TensorDef(T, S.M, S.N, S.K),
                     O=TensorDef(T, S.M, S.K, S.N, output=True)):
  domain(D.m, D.k, D.n)
  O[D.m, D.k, D.n] = I[D.m, D.n, D.k]


@linalg_structured_op
def transpose_3d_102(I=TensorDef(T, S.M, S.N, S.K),
                     O=TensorDef(T, S.N, S.M, S.K, output=True)):
  domain(D.n, D.m, D.k)
  O[D.n, D.m, D.k] = I[D.m, D.n, D.k]


@linalg_structured_op
def transpose_3d_120(I=TensorDef(T, S.M, S.N, S.K),
                     O=TensorDef(T, S.N, S.K, S.M, output=True)):
  domain(D.n, D.k, D.m)
  O[D.n, D.k, D.m] = I[D.m, D.n, D.k]


@linalg_structured_op
def transpose_3d_201(I=TensorDef(T, S.M, S.N, S.K),
                     O=TensorDef(T, S.K, S.M, S.N, output=True)):
  domain(D.k, D.m, D.n)
  O[D.k, D.m, D.n] = I[D.m, D.n, D.k]


@linalg_structured_op
def transpose_3d_210(I=TensorDef(T, S.M, S.N, S.K),
                     O=TensorDef(T, S.K, S.N, S.M, output=True)):
  domain(D.k, D.n, D.m)
  O[D.k, D.n, D.m] = I[D.m, D.n, D.k]


@linalg_structured_op
def transpose_4d_0123(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.M, S.N, S.K, S.L, output=True)):
  domain(D.m, D.n, D.k, D.l)
  O[D.m, D.n, D.k, D.l] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_0132(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.M, S.N, S.L, S.K, output=True)):
  domain(D.m, D.n, D.l, D.k)
  O[D.m, D.n, D.l, D.k] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_0213(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.M, S.K, S.N, S.L, output=True)):
  domain(D.m, D.k, D.n, D.l)
  O[D.m, D.k, D.n, D.l] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_0231(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.M, S.K, S.L, S.N, output=True)):
  domain(D.m, D.k, D.l, D.n)
  O[D.m, D.k, D.l, D.n] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_0312(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.M, S.L, S.N, S.K, output=True)):
  domain(D.m, D.l, D.n, D.k)
  O[D.m, D.l, D.n, D.k] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_0321(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.M, S.L, S.K, S.N, output=True)):
  domain(D.m, D.l, D.k, D.n)
  O[D.m, D.l, D.k, D.n] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_1023(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.N, S.M, S.K, S.L, output=True)):
  domain(D.n, D.m, D.k, D.l)
  O[D.n, D.m, D.k, D.l] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_1032(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.N, S.M, S.L, S.K, output=True)):
  domain(D.n, D.m, D.l, D.k)
  O[D.n, D.m, D.l, D.k] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_1203(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.N, S.K, S.M, S.L, output=True)):
  domain(D.n, D.k, D.m, D.l)
  O[D.n, D.k, D.m, D.l] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_1230(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.N, S.K, S.L, S.M, output=True)):
  domain(D.n, D.k, D.l, D.m)
  O[D.n, D.k, D.l, D.m] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_1302(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.N, S.L, S.M, S.K, output=True)):
  domain(D.n, D.l, D.m, D.k)
  O[D.n, D.l, D.m, D.k] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_1320(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.N, S.L, S.K, S.M, output=True)):
  domain(D.n, D.l, D.k, D.m)
  O[D.n, D.l, D.k, D.m] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_2013(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.K, S.M, S.N, S.L, output=True)):
  domain(D.k, D.m, D.n, D.l)
  O[D.k, D.m, D.n, D.l] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_2031(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.K, S.M, S.L, S.N, output=True)):
  domain(D.k, D.m, D.l, D.n)
  O[D.k, D.m, D.l, D.n] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_2103(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.K, S.N, S.M, S.L, output=True)):
  domain(D.k, D.n, D.m, D.l)
  O[D.k, D.n, D.m, D.l] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_2130(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.K, S.N, S.L, S.M, output=True)):
  domain(D.k, D.n, D.l, D.m)
  O[D.k, D.n, D.l, D.m] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_2301(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.K, S.L, S.M, S.N, output=True)):
  domain(D.k, D.l, D.m, D.n)
  O[D.k, D.l, D.m, D.n] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_2310(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.K, S.L, S.N, S.M, output=True)):
  domain(D.k, D.l, D.n, D.m)
  O[D.k, D.l, D.n, D.m] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_3012(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.L, S.M, S.N, S.K, output=True)):
  domain(D.l, D.m, D.n, D.k)
  O[D.l, D.m, D.n, D.k] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_3021(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.L, S.M, S.K, S.N, output=True)):
  domain(D.l, D.m, D.k, D.n)
  O[D.l, D.m, D.k, D.n] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_3102(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.L, S.N, S.M, S.K, output=True)):
  domain(D.l, D.n, D.m, D.k)
  O[D.l, D.n, D.m, D.k] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_3120(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.L, S.N, S.K, S.M, output=True)):
  domain(D.l, D.n, D.k, D.m)
  O[D.l, D.n, D.k, D.m] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_3201(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.L, S.K, S.M, S.N, output=True)):
  domain(D.l, D.k, D.m, D.n)
  O[D.l, D.k, D.m, D.n] = I[D.m, D.n, D.k, D.l]


@linalg_structured_op
def transpose_4d_3210(I=TensorDef(T, S.M, S.N, S.K, S.L),
                      O=TensorDef(T, S.L, S.K, S.N, S.M, output=True)):
  domain(D.l, D.k, D.n, D.m)
  O[D.l, D.k, D.n, D.m] = I[D.m, D.n, D.k, D.l]
