from typing import Sequence

from argparse import ArgumentParser
import nevergrad as ng
import numpy as np
from typing import Callable, Sequence

debug_constraints = False


# Search tile sizes must either be 0 or divide the problem size.
def constraint_all_must_divide(problem_sizes: Sequence[int],
                               search_sizes: Sequence[int]):
  if debug_constraints:
    print(f'C1 problem_sizes:{problem_sizes} vs search_sizes:{search_sizes}')

  for size, search_size in zip(problem_sizes, search_sizes):
    if search_size != 0 and size % search_size != 0:
      return False
  return True


# Search sizes must not yield too much unrolling.
def constraint_unrolling_not_too_big(problem_sizes: Sequence[int],
                                     search_sizes: Sequence[int],
                                     unrolling_limit: int):
  if debug_constraints:
    print(f'C2 problem_sizes:{problem_sizes} vs search_sizes:{search_sizes}')

  prod = 1
  for size, search_size in zip(problem_sizes, search_sizes):
    prod = prod * size if search_size == 0 else prod * search_size
  if debug_constraints:
    print(f'C2 prod {prod}')
  return prod < unrolling_limit


# Search sizes must be smaller than the problem size.
def constraint_in_bounds(problem_sizes: Sequence[int],
                         search_sizes: Sequence[int]):
  if debug_constraints:
    print(f'C3 problem_sizes:{problem_sizes} vs search_sizes:{search_sizes}')

  for size, search_size in zip(problem_sizes, search_sizes):
    if search_size > size:
      return False
  return True


# Volume must not be smaller than both a limit and a fraction of problem size.
def constraint_volume_not_too_small(
    problem_sizes: Sequence[int], search_sizes: Sequence[int],
    volume_limit: int, volume_relative_percentage_lower_bound: int):
  if debug_constraints:
    print(f'C4 problem_sizes:{problem_sizes} vs search_sizes:{search_sizes}')

  size_prod = np.prod(problem_sizes)
  search_size_prod = 1
  for size, search_size in zip(problem_sizes, search_sizes):
    search_size_prod = search_size_prod * size \
      if search_size == 0 else search_size_prod * search_size

  return search_size_prod >= volume_limit or \
    100 * search_size_prod + 1 > volume_relative_percentage_lower_bound * size_prod


# Entry point to combine all size constraints.
def size_constraints_conjunction_satisfied(
    problem_sizes: Sequence[int],
    search_sizes: Sequence[int],
    unrolling_limit: int = 10000,
    volume_limit: int = 16,
    volume_relative_percentage_lower_bound: int = 25):
  return constraint_all_must_divide(problem_sizes, search_sizes) \
    and constraint_unrolling_not_too_big(problem_sizes,   \
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


# State that is shared across parent and children processes in a MP run.
# All information within this state must "pickle".
class IPCState:

  def __init__(self, success: bool, throughputs: Sequence, problem=None):
    self.success = success
    self.throughputs = throughputs
    self.problem = problem


class NGExecution:

  def __init__(self, proposal, problem_instance):
    self.proposal = proposal
    self.problem_instance = problem_instance


class NGMPExecution(NGExecution):

  def __init__(self, proposal, problem_instance, process, ipc_dict):
    super().__init__(proposal, problem_instance)
    self.process = process
    self.ipc_dict = ipc_dict

  def future_ipc_state(self):
    return self.ipc_dict['result']


# Root process sleeps and synchronizes all forked bulk_synchronous processes.
# Additionally, this updates the NGMPExecutions accordingly for further
# processing across the single/multi process boundary:
#   1. on a timeout or an exception, the corresponding entry in processes and
#      problem_instances is set to None to avoid touching them again in the
#      future.
#   2. on a successful finish, the corresponding entry in problem_instances is
#      set to the corresponding ipc_dict['problem']. This is used to
#      communicated back the compiled state of the problem across the single/multi
#      process boundary.
def join_bulk_synchronous_processes(ng_mp_executions: Sequence[NGMPExecution],
                                    timeout: float):
  import time
  remaining_time = timeout
  found_alive = True
  joined_with_root = [False for p in ng_mp_executions]
  # Busy-wait to implement a simple polling mechanism.
  while found_alive and remaining_time > 0:
    found_alive = False
    remaining_time = remaining_time - 0.1
    time.sleep(0.1)
    for idx in range(len(ng_mp_executions)):
      # This process was already joined, skip.
      if joined_with_root[idx] is True:
        continue

      process = ng_mp_executions[idx].process
      # This process finished by itself, join with it and mark it joined.
      if not process.is_alive():
        process.join()
        joined_with_root[idx] = True
        continue

      # Otherwise, we found an alive process, continue until timer expires.
      found_alive = True

  # Timer is up, any alive processes now need to be terminated.
  for joined, execution in zip(joined_with_root, ng_mp_executions):
    if joined:
      continue
    print(f'timeout: {execution.proposal} did not complete within {timeout}s')
    execution.process.terminate()
    execution.process.join()
    # Override to return a failed IPCState and signify infinite relative_error.
    execution.future_ipc_state = lambda: IPCState(success=False,
                                                  throughputs=None)


# Add tuning-specific arguments to the parser.
def add_argparser_tuning_arguments(parser: ArgumentParser):
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
      nargs='+',
      default=['RandomSearch'],
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
