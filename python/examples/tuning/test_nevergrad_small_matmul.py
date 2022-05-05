from argparse import ArgumentParser
import multiprocessing as mp
import nevergrad as ng
import numpy as np

import mlir.iree_sandbox as sandbox
import mlir.ir as ir
import mlir.dialects.pdl as pdl
import mlir.dialects.iree_linalg_transform as transform

from ..core.experts import *
from ..core.harness import *
from ..core.nevergrad_parallel_utils import *
from ..core.nevergrad_tuner_utils import *
from ..core import nevergrad_searchable_strategies as strategies
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

  def __init__(self, pb_sizes: Sequence[int]):
    self.problem_sizes = pb_sizes

    # Init nevergrad's instrumentation.
    # Note: ts=[6, 16, 1], interchange=[0, 2, 1] can reach 186GFl/s (out of 192).
    self.register_tile_sizes = strategies.SearchableList( \
        strategies.IntChoice('register_ts0')
          .with_range(range(1, min(32, pb_sizes[0])))
          .with_values([1,4,5,6,7,8]),
        strategies.IntChoice('register_ts1')
          .with_range(range(1, min(32, pb_sizes[1])))
          .with_values([1,4,6,8,12,16,32]),
        strategies.IntChoice('register_ts2')
          .with_range(range(1, min(32, pb_sizes[2])))
          .with_values([1, 8, 16, 32]),
    )
    self.register_interchange = strategies.Permutation(
        'register_interchange', length=3)\
          .with_permutation_subset([[0, 2, 1], [0, 1, 2], [1, 2, 0], [1, 0, 2]])
    self.register_peel = strategies.SearchableList( \
        strategies.BoolChoice('register_peel', length=3)
    )
    # TODO: atm there are some issues with multi-handles after peeling and
    # passing that to scalarize_dyn_dims, so we force that part of search to be
    # False.
    self.register_scalarize_dyn_dims = strategies.BoolChoice(
        'register_scalarize_dyn_dims')
    self.tuning_knobs = strategies.SearchableList( \
      self.register_tile_sizes,
      self.register_interchange,
      self.register_peel,
      self.register_scalarize_dyn_dims
    )
    self.instrumentation = ng.p.Instrumentation(
        **self.tuning_knobs.get_instrumentation())

  def build_compile_time_problem_sizes(self):
    return {k: v for k, v in zip(keys, self.problem_sizes)}

  def build_compile_time_elemental_problem_types(self):
    return {k: v for k, v in zip(keys, self.problem_sizes)}

  # TODO: more advanced schedules, atm we just TileAndVectorize + peel.
  def schedule(self, module, proposal, benefit: int = 1):
    print(f'Problem sizes: {self.problem_sizes}')
    print(
        f'Register tile sizes: {self.register_tile_sizes.extract_from_proposal(proposal)}'
    )
    print(f'Register interchange:' +
          f' {self.register_interchange.extract_from_proposal(proposal)}')
    print(f'Register peel:' +
          f' {self.register_peel.extract_from_proposal(proposal)}')
    print(
        f'Register scalarize dyn dims:' +
        f' {self.register_scalarize_dyn_dims.extract_from_proposal(proposal)}')

    with InsertionPoint(module.body):
      # TODO: Evolve to python-metaprogrammed PDLL constraints.
      matcher = match_op_with_sizes_multiple_of(
          module, equivalent_op_name='linalg.matmul')
      sequence = transform.SequenceOp()
      with ir.InsertionPoint(sequence.body.blocks[0]) as ip:
        tile_strategy = strategies.Tile(self.register_tile_sizes,
                                        self.register_interchange,
                                        self.register_peel,
                                        self.register_scalarize_dyn_dims)
        tile_strategy.build_transform_ir_under_insertion_point(\
          target=transform.MatchOp(matcher),
          proposal=proposal)

        transform.VectorizeOp()
        strategies.lowering_transform_ir_under_insertion_point()


def make_optimizer(scheduler: NGSchedulerInterface, search_strategy: str,
                   budget: int):
  optimizer = ng.optimizers.registry[search_strategy](
      parametrization=scheduler.instrumentation, budget=budget)
  return optimizer


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
  scheduler = NGScheduler(parsed_args.problem_sizes_list[0])

  optimizer = make_optimizer(scheduler, parsed_args.search_strategy,
                             parsed_args.search_budget)

  async_optim_loop(problem_definition, scheduler, optimizer, parsed_args)


if __name__ == '__main__':
  main()
