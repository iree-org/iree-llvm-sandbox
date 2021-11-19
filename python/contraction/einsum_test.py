# RUN: %PYTHON %s 2>&1 | FileCheck %s

# Check that various einsum specifications can actually be compiled. This is not
# a performance benchmark so we are not doing transformations other than
# lowering.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import *

dummy_expert = TransformationList(transforms=[Bufferize()] +
                                  StagedLowerVectorsTransformationList() +
                                  [LowerToLLVM()])


def main():
  problem_definition = EinsumProblem("klnp,nk->pl")
  problem = ProblemInstance(problem_definition, problem_definition.keys(),
                            [np.float32] * 3)
  sizes = {k: v for k, v in zip(problem_definition.keys(), [10, 12, 14, 16])}
  problem.compile(
      entry_point_name="einsum_main",
      fun_to_benchmark_name="einsum_on_tensors",
      compile_time_problem_sizes_dict=sizes,
      transform=dummy_expert)
  problem.run(
      n_iters=1,
      entry_point_name="einsum_main",
      runtime_problem_sizes_dict=sizes)


if __name__ == "__main__":
  main()
