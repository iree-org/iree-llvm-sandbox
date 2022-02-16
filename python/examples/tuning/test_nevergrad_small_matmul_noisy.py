from argparse import ArgumentParser
import multiprocessing as mp
import nevergrad as ng
import numpy as np

import mlir.iree_sandbox as sandbox
import mlir.ir as ir
import mlir.dialects.pdl as pdl
import mlir.dialects.linalg_transform as transform

from ..core.experts import *
from ..core.harness import *
from ..core.nevergrad_parallel_utils import *
from ..core.nevergrad_tuner_utils import *
from ..core.pdl_utils import *
from ..core.problem_definition import ProblemDefinition
from ..core.transforms import *
from ..core.utils import *

from ..contraction.definitions import EinsumProblem

################################################################################
### Problem instantiations.
################################################################################

keys = ['m', 'n', 'k']

# CHECK-NOT: FAILURE


# TODO: Generalize and reduce this class.
class NGScheduler(NGSchedulerInterface):

  def __init__(self, \
               problem_definition: ProblemDefinition,
               problem_sizes: Sequence[int]):
    # TODO: better factoring of these constants.
    self.entry_point_name = 'main'
    self.fun_to_benchmark_name = 'matmul'

    self.problem_definition = problem_definition
    self.problem_sizes = problem_sizes

    # Dictionary of matcher names.
    self.matchers = {}

    # Init nevergrad's instrumentation.
    # TODO: better parametrization, for now this is enough to get a prototype.
    # TODO: atm we remove 0 from search because peeling freaks out if we don't
    # tile but still specify a non-existent loop to peel.
    self.register_tile_sizes = ng.p.Choice(list(range(1, 33, 1)), repetitions=3)
    self.register_interchange = ng.p.Choice(list(range(6)))
    self.tile_partial_by_one = ng.p.Choice([False, True], repetitions=3)

    self.register_tile_sizes_search_keyword = 'register_tile_sizes'
    self.register_interchange_search_keyword = 'register_interchange'
    self.tile_partial_by_one_search_keyword = 'tile_partial_by_one'
    self.instrumentation = ng.p.Instrumentation(
        register_tile_sizes=self.register_tile_sizes,
        register_interchange=self.register_interchange,
        tile_partial_by_one=self.tile_partial_by_one)
    self.optimizer = None

  def build_compile_time_problem_sizes(self):
    return {k: v for k, v in zip(keys, self.problem_sizes)}

  def build_compile_time_elemental_problem_types(self):
    return {k: v for k, v in zip(keys, self.problem_sizes)}

  def set_optimizer(self, search_strategy: str, budget: int):

    def constraints_fun(proposal):
      return dispatch_size_constraints_conjunction_satisfied( \
        self.problem_sizes, proposal, self.register_tile_sizes_search_keyword)

    self.search_strategy = search_strategy
    self.optimizer = ng.optimizers.registry[self.search_strategy](
        parametrization=self.instrumentation, budget=budget)
    self.optimizer.parametrization.register_cheap_constraint(constraints_fun)

  # Unwrap the np.array from NG's ask() kwargs.
  def extract_register_tile_sizes_from_proposal(self, proposal):
    return [x for x in proposal.kwargs[self.register_tile_sizes_search_keyword]]

  def extract_register_interchange_from_proposal(self, proposal):
    permutations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1],
                    [2, 1, 0]]
    return permutations[proposal.kwargs[
        self.register_interchange_search_keyword]]

  def extract_tile_partial_by_one_from_proposal(self, proposal):
    return [x for x in proposal.kwargs[self.tile_partial_by_one_search_keyword]]

  # Optimizer may want to override our constraints set.
  def validate_proposal(self, proposal):
    if not size_constraints_conjunction_satisfied(
        self.problem_sizes,
        self.extract_register_tile_sizes_from_proposal(proposal)):
      return False
    return True

  # TODO: Evolve to python-metaprogrammed PDLL constraints.
  def create_matchers(self, module, benefit: int = 1):
    #                  M=A.0   N=B.1   K=A.1
    dimM, dimN, dimK = [0, 0], [1, 1], [0, 1]
    self.matchers['isa_linalg_matmul'] = \
      match_op_with_sizes_multiple_of(
        module, equivalent_op_name='linalg.matmul')
    self.matchers['isa_linalg_matmul_d_x_x'] = \
      match_op_with_dynamic_or_static_sizes(
            module,
            equivalent_op_name='linalg.matmul',
            dynamic_spec_list=['d'],
            op_dim_spec_list=[dimM])
    self.matchers['isa_linalg_matmul_s_d_x'] = \
      match_op_with_dynamic_or_static_sizes(
            module,
            equivalent_op_name='linalg.matmul',
            dynamic_spec_list=['s', 'd'],
            op_dim_spec_list=[dimM, dimN])
    self.matchers['isa_linalg_matmul_s_s_d'] = \
      match_op_with_dynamic_or_static_sizes(
            module,
            equivalent_op_name='linalg.matmul',
            dynamic_spec_list=['s', 's', 'd'],
            op_dim_spec_list=[dimM, dimN, dimK])
    self.matchers['isa_linalg_matmul_s_s_s'] = \
      match_op_with_dynamic_or_static_sizes(
            module,
            equivalent_op_name='linalg.matmul',
            dynamic_spec_list=['s', 's', 's'],
            op_dim_spec_list=[dimM, dimN, dimK])

  # TODO: more advanced schedules, atm we just TileAndVectorize + peel.
  def schedule(self, module, proposal, benefit: int = 1):
    register_tile_sizes = self.extract_register_tile_sizes_from_proposal(
        proposal)
    register_interchange = self.extract_register_interchange_from_proposal(
        proposal)
    tile_partial_by_one = self.extract_tile_partial_by_one_from_proposal(
        proposal)

    print(f'Problem sizes: {self.problem_sizes}')
    print(f'Register tile sizes: {register_tile_sizes}')
    print(f'Register interchange: {register_interchange}')
    print(f'Tile partial by one: {tile_partial_by_one}')

    # TODO: this is necessary to force-load the dialect, otherwise op creation
    # complains about "unregistered dialect" despite the registration being called.
    register_sandbox_passes_and_dialects(module.context)
    module.context.dialects["linalg_transform"]

    with InsertionPoint(module.body):
      self.create_matchers(module, benefit)
      sequence = transform.SequenceOp()
      with ir.InsertionPoint(sequence.body.blocks[0]):
        matched = transform.MatchOp(self.matchers['isa_linalg_matmul'])
        transform.TileOp(matched,
                         sizes=register_tile_sizes,
                         interchange=register_interchange,
                         peel=[0, 1, 2])

        if tile_partial_by_one[0]:
          matched = transform.MatchOp(self.matchers['isa_linalg_matmul_d_x_x'])
          transform.TileOp(matched, sizes=[1])

        if tile_partial_by_one[1]:
          matched = transform.MatchOp(self.matchers['isa_linalg_matmul_s_d_x'])
          transform.TileOp(matched, sizes=[0, 1])

        if tile_partial_by_one[2]:
          matched = transform.MatchOp(self.matchers['isa_linalg_matmul_s_s_d'])
          transform.TileOp(matched, sizes=[0, 0, 1])

        matched = transform.MatchOp(self.matchers['isa_linalg_matmul_s_s_s'])
        transform.VectorizeOp(matched)

        transform.BufferizeOp()
        for i in range(7):
          transform.LowerVectorsOp(stages=list(j + 1 for j in range(i + 1)))

        transform.LowerToLLVMOp()


def main():
  argparser = ArgumentParser()
  add_argparser_arguments(argparser, default_problem_sizes_list=[[16, 16, 16]])
  add_argparser_tuning_arguments(argparser)
  parsed_args = argparser.parse_args()

  # Sanity checks before starting.
  assert len(
      parsed_args.problem_sizes_list) == 1, 'Single problem size supported atm'

  # Init random seed for reproducibility.
  np.random.seed(parsed_args.random_seed)

  problem_definition = EinsumProblem('mk,kn', 'mnk', 2)
  # Create a schedule builder for fixed sizes.
  scheduler = NGScheduler(\
    problem_definition = problem_definition,
    problem_sizes = parsed_args.problem_sizes_list[0])

  scheduler.set_optimizer(parsed_args.search_strategy,
                          parsed_args.search_budget)

  async_optim_loop(problem_definition, scheduler, parsed_args)


if __name__ == '__main__':
  main()
