import typing as tp

from argparse import ArgumentParser
import numpy as np
from typing import Callable, Sequence

import mlir.ir as ir

debug_constraints = False


def save_module(module, module_save_filename):
  """Helper function to save a module to a file."""

  with open(module_save_filename, 'w') as f:
    f.write(str(module))
  print(f'Module saved in {module_save_filename}')


class NGSchedulerInterface:
  """Generic interface for schedule search via nevergrad."""

  def build_compile_time_problem_sizes(self):
    """Build the dictionary of (dimension_name, size) giving the sizes to compile"""
    pass

  def set_optimizer(self, search_strategy: str, budget: int):
    """Set up the nevergrad optimizer, with its instrumentation and constraints"""
    pass

  def validate_proposal(self, proposal) -> bool:
    """Validate a proposal made by the optimizer.
    
    This handles the case where our search space is too sparse and the optimizer
    may want to override our constraints set.
    """
    pass

  def create_matchers(self, module, benefit: int = 1):
    """Create the PDL matchers 
    
    Create the PDL IR for matchers / constraints inside `module`.
    """
    pass

  def schedule(self, module, proposal, benefit: int = 1):
    """Create the PDL matchers 
    
    Create the PDL IR for schedule inside `module`.
    """
    pass

  def save_proposal_as_module(self,
                              proposal,
                              module_save_filename,
                              benefit: int = 1):
    with ir.Context() as ctx, ir.Location.unknown() as loc:
      module = ir.Module.create()
      self.schedule(module, proposal, benefit)
      save_module(module, module_save_filename)
      with open(module_save_filename, 'w') as f:
        f.write(str(module))


################################################################################
### Nevergrad constraints.
### TODO: somehow connect to PDL matchers.
################################################################################


def constraint_all_must_divide(problem_sizes: Sequence[int],
                               search_sizes: Sequence[int]):
  """Constraint to specify `search_sizes` are either 0 or divide `problem_sizes`."""

  if debug_constraints:
    print(f'C1 problem_sizes:{problem_sizes} vs search_sizes:{search_sizes}')

  for size, search_size in zip(problem_sizes, search_sizes):
    if search_size != 0 and size % search_size != 0:
      return False
  return True


def constraint_unrolling_not_too_big(problem_sizes: Sequence[int],
                                     search_sizes: Sequence[int],
                                     unrolling_limit: int):
  """Constraint to specify `search_sizes` do not yield too much unrolling."""

  if debug_constraints:
    print(f'C2 problem_sizes:{problem_sizes} vs search_sizes:{search_sizes}')

  prod = 1
  for size, search_size in zip(problem_sizes, search_sizes):
    prod = prod * size if search_size == 0 else prod * search_size
  if debug_constraints:
    print(f'C2 prod {prod}')
  return prod < unrolling_limit


def constraint_in_bounds(problem_sizes: Sequence[int],
                         search_sizes: Sequence[int]):
  """Constraint to limit `search_sizes` to the extent of the problem_sizes."""

  if debug_constraints:
    print(f'C3 problem_sizes:{problem_sizes} vs search_sizes:{search_sizes}')

  for size, search_size in zip(problem_sizes, search_sizes):
    if search_size > size:
      return False
  return True


def constraint_volume_not_too_small(
    problem_sizes: Sequence[int], search_sizes: Sequence[int],
    volume_limit: int, volume_relative_percentage_lower_bound: int):
  """Constraint to skip `search_sizes` that would yield a too small volume."""

  if debug_constraints:
    print(f'C4 problem_sizes:{problem_sizes} vs search_sizes:{search_sizes}')

  size_prod = np.prod(problem_sizes)
  search_size_prod = 1
  for size, search_size in zip(problem_sizes, search_sizes):
    search_size_prod = search_size_prod * size \
      if search_size == 0 else search_size_prod * search_size

  return search_size_prod >= volume_limit or \
    100 * search_size_prod + 1 > volume_relative_percentage_lower_bound * size_prod


def size_constraints_conjunction_satisfied(
    problem_sizes: Sequence[int],
    search_sizes: Sequence[int],
    unrolling_limit: int = 10000,
    volume_limit: int = 16,
    volume_relative_percentage_lower_bound: int = 25):
  """Constraint to perform the conjunction of known constraints."""

  return constraint_unrolling_not_too_big(problem_sizes,   \
                                          search_sizes,    \
                                          unrolling_limit) \
      and constraint_volume_not_too_small(problem_sizes, \
                                          search_sizes,  \
                                          volume_limit,  \
                                          volume_relative_percentage_lower_bound) \
        and constraint_in_bounds(problem_sizes, search_sizes)


# A `proposal` coming from Nevergrad contains a tuple of (*args, **kwargs)
# we want to extract the search sizes and we need to unwrap the np.array to
# a list.
def dispatch_size_constraints_conjunction_satisfied(
    problem_sizes: Sequence[int], proposal, keyword):
  # Unwrap the np.array
  proposed_search_sizes = [x for x in proposal[1][keyword]]
  return size_constraints_conjunction_satisfied(problem_sizes,
                                                proposed_search_sizes)


################################################################################
### Argparser
################################################################################
def add_argparser_tuning_arguments(parser: ArgumentParser):
  """Add tuning-specific arguments to the parser."""

  parser.add_argument('--machine-peak', type=int, nargs='?', default=192)
  parser.add_argument('--metric-to-measure',
                      type=str,
                      nargs='?',
                      default='gflop_per_s_per_iter')
  parser.add_argument(
      '--output-dir',
      type=str,
      nargs='?',
  )
  parser.add_argument('--num-compilation-processes',
                      type=int,
                      nargs='?',
                      default=1)
  parser.add_argument('--random-seed', type=int, nargs='?', default=42)
  parser.add_argument('--search-budget', type=int, nargs='?', default=100)
  parser.add_argument(
      '--search-strategy',
      type=str,
      nargs='?',
      default='RandomSearch',
  )
  parser.add_argument('--timeout-per-compilation',
                      type=int,
                      nargs='?',
                      default=5)
  # Until ExecutionEngine pickles, we are both compiling and evaluating.
  parser.add_argument('--timeout-per-evaluation',
                      type=int,
                      nargs='?',
                      default=5)
