# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains test to compile fusion examples.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from .definitions import *

################################################################################
### Expert for running the fusion tests.
################################################################################


class FusionTestExpert(TransformationList):

  def __init__(self, fn_name: str, root_op_name: str, tile_sizes: Sequence[int],
               tile_interchange: Sequence[int],
               vectorize_op_list: Sequence[str], **kwargs):
    t = [
        Fuse(
            fn_name,
            root_op_name,
            tile_sizes=tile_sizes,
            tile_interchange=tile_interchange,
            pad=False)
    ]
    for vectorize_op_name in vectorize_op_list:
      t += [Vectorize(fn_name, vectorize_op_name)]
    t += [Bufferize(), Print()
          ] + [LowerVectors(stage=i) for i in range(7)] + [LowerToLLVM()]
    d = {'transforms': t}
    kwargs.update(d)
    TransformationList.__init__(self, **kwargs)


# 1 linalg.fill -> linalg.matmul fusion.
def fill_matmul_fusion():

  expert = FusionTestExpert(
      fn_name='matmul_on_tensors',
      root_op_name='linalg.matmul',
      tile_sizes=[4, 8, 6],
      tile_interchange=[0, 1, 2],
      vectorize_op_list=['linalg.matmul', 'linalg.fill'])
  problem_sizes_dict = {'M': 24, 'N': 32, 'K': 48}
  problem = ProblemInstance(
      problem_definition=MatmulProblem(),
      np_types=[np.float32, np.float32, np.float32])

  ## These lit tests are not actually run, but can be checked locally using
  ## $ bazel run ${IREE_LLVM_SANDBOX_DIR}:fusion_test | \
  ##     bazel run ${LLVM_DIR}/llvm:FileCheck ${IREE_LLVM_SANDBOX_DIR}/python/fusion/test.py

  #      CHECK: func @matmul_on_tensors(
  # CHECK-SAME:     %[[ARG0:.+]]: memref<24x48xf32>
  # CHECK-SAME:     %[[ARG1:.+]]: memref<48x32xf32>
  # CHECK-SAME:     %[[ARG2:.+]]: memref<24x32xf32>)
  #      CHECK:   %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : vector<4x8xf32>
  #      CHECK:   scf.for %{{.+}} =
  #      CHECK:     scf.for %{{.+}} =
  #      CHECK:       %[[REDUCTION:.+]] = scf.for %[[IV1:[a-zA-Z0-9]+]]
  # CHECK-SAME:           iter_args(%[[PHI:.+]] = %[[ZERO]]
  #      CHECK:         %[[LHS_VEC:.+]] = vector.transfer_read %[[ARG0]]
  #      CHECK:         %[[RHS_VEC:.+]] = vector.transfer_read %[[ARG1]]
  #      CHECK:         %[[CONTRACT:.+]] = vector.contract
  # CHECK-SAME:            %[[LHS_VEC]], %[[RHS_VEC]], %[[PHI]]
  #      CHECK:          scf.yield %[[CONTRACT]]
  #      CHECK:       vector.transfer_write %[[REDUCTION]], %[[ARG2]]
  problem.compile(
      entry_point_name='matmul_main',
      fun_to_benchmark_name='matmul_on_tensors',
      compile_time_problem_sizes_dict=problem_sizes_dict,
      transform=expert)


def fill_matmul_bias_add_fusion():
  expert = FusionTestExpert(
      fn_name='matmul_bias_add_on_tensors',
      root_op_name='linalg.generic',
      tile_sizes=[4, 8, 6],
      tile_interchange=[0, 1, 2],
      vectorize_op_list=['linalg.matmul', 'linalg.fill', 'linalg.generic'])
  problem_sizes_dict = {'M': 24, 'N': 32, 'K': 48}
  problem = ProblemInstance(
      problem_definition=MatmulBiasAddProblem(),
      np_types=[np.float32, np.float32, np.float32, np.float32])

  ## These lit tests are not actually run, but can be checked locally using
  ## $ bazel run ${IREE_LLVM_SANDBOX_DIR}:fusion_test | \
  ##     bazel run ${LLVM_DIR}/llvm:FileCheck ${IREE_LLVM_SANDBOX_DIR}/python/fusion/test.py
  #
  #      CHECK: func @matmul_bias_add_on_tensors(
  # CHECK-SAME:     %[[ARG0:.+]]: memref<24x48xf32>
  # CHECK-SAME:     %[[ARG1:.+]]: memref<48x32xf32>
  # CHECK-SAME:     %[[ARG2:.+]]: memref<32xf32>
  # CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: memref<24x32xf32>
  # CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: memref<24x32xf32>)
  #      CHECK:   scf.for %{{.+}} =
  #      CHECK:     %[[LHS_VEC:.+]] = vector.transfer_read %[[ARG0]]
  #      CHECK:     scf.for %{{.+}} =
  #      CHECK:       %[[RHS_VEC:.+]] = vector.transfer_read %[[ARG1]]
  #      CHECK:       %[[INIT_VEC:.+]] = vector.transfer_read %[[ARG2]]
  #      CHECK:       %[[BCAST:.+]] = vector.broadcast %[[INIT_VEC]]
  #      CHECK:       %[[CONTRACT:.+]] = vector.contract
  # CHECK-SAME:          %[[LHS_VEC]], %[[RHS_VEC]], %[[BCAST]]
  #      CHECK:       vector.transfer_write %[[CONTRACT]], %[[ARG4]]
  problem.compile(
      entry_point_name='matmul_bias_add_main',
      fun_to_benchmark_name='matmul_bias_add_on_tensors',
      compile_time_problem_sizes_dict=problem_sizes_dict,
      transform=expert)


def main():
  fill_matmul_fusion()
  fill_matmul_bias_add_fusion()

if __name__ == "__main__":
  main()
