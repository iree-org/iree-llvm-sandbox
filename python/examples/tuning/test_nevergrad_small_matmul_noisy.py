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
from ..core.nevergrad_tuner_utils import *
from ..core.pdl_utils import *
from ..core.transforms import *
from ..core.utils import *

from ..contraction.definitions import EinsumProblem

################################################################################
### Problem instantiations.
################################################################################

keys = ['m', 'n', 'k']

# CHECK-NOT: FAILURE


# TODO: improve this to also generate good matcher constraints.
class NGScheduler:

  def __init__(self, problem_definition: ProblemDefinition,
               problem_sizes: Sequence[int]):
    # TODO: better factoring of these constants
    self.entry_point_name = 'main'
    self.fun_to_benchmark_name = 'matmul'

    self.problem_definition = problem_definition
    self.problem_sizes = problem_sizes

    # Dictionary of matcher names.
    self.matchers = {}

    # Init nevergrad's instrumentation.
    # TODO: better parametrization, for now this is enough to get a prototype.
    # TODO: atm we remove 0 from search because peeling freaks out if we don't
    # tile and still specify a non-existent loop to
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

  # TODO: generate a unique name for the matcher.
  # TODO: generate a tight matcher for the generic (e.g. isMatmulOp with
  # fixed size).
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

  # TODO: more advanced schedules, atm we just TileAndVectorize.
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

  def save_module(self, module, module_save_filename):
    with open(module_save_filename, 'w') as f:
      f.write(str(module))
    print(f'Module saved in {module_save_filename}')

  def save_proposal_as_module(self,
                              proposal,
                              module_save_filename,
                              benefit: int = 1):
    with ir.Context() as ctx, ir.Location.unknown() as loc:
      module = Module.create()
      self.schedule(module, proposal, benefit)
      self.save_module(module, module_save_filename)


# Entry point to try compile and run while catching and reporting exceptions.
# This can run in interruptible multiprocess mode.
# ipc_dict must be provided, and it is used to return information across the
# root/children process boundary:
#   - 'throughputs': the measured throughputs.
#   - 'success': the return status.
def compile_and_run_checked_mp(problem: ProblemInstance, scheduler: NGScheduler,
                               proposal, n_iters: int, ipc_dict: dict):
  try:
    # Construct the schedule and save the module in case we need to replay later.
    def schedule_and_save(module):
      scheduler.schedule(module, proposal)
      # TODO: save and report on error.

    problem.compile_with_schedule_builder(
        entry_point_name=scheduler.entry_point_name,
        fun_to_benchmark_name=scheduler.fun_to_benchmark_name,
        compile_time_problem_sizes_dict={
            k: v for k, v in zip(keys, scheduler.problem_sizes)
        },
        schedule_builder=schedule_and_save)

    throughputs = problem.run(
        n_iters=n_iters,
        entry_point_name=scheduler.entry_point_name,
        runtime_problem_sizes_dict=problem.compile_time_problem_sizes_dict)

    ipc_dict['result'] = IPCState(success=True, throughputs=throughputs)
  except Exception as e:
    import traceback
    traceback.print_exc()
    # TODO: save to replay errors.
    print(e)
    ipc_dict['result'] = IPCState(success=False, throughputs=None)


def bulk_synchronous_optim_loop(parsed_args):
  # Init random seed for reproducibility.
  np.random.seed(parsed_args.random_seed)

  assert len(
      parsed_args.problem_sizes_list) == 1, 'Single problem size supported atm'

  # Sanity check so we can pin one process per CPU core and avoid too much noise.
  assert parsed_args.num_compilation_processes < os.cpu_count(), \
    f'Num compilation processes: {parsed_args.num_compilation_processes}' + \
    f' cannot run on num CPUs: {os.cpu_count()}'

  problem_definition = EinsumProblem('mk,kn', 'mnk', 2)
  # Create a schedule builder for fixed sizes.
  scheduler = NGScheduler(\
    problem_definition = problem_definition,
    problem_sizes = parsed_args.problem_sizes_list[0])

  manager = mp.Manager()
  for search_strategy in parsed_args.search_strategy:
    scheduler.set_optimizer(search_strategy, parsed_args.search_budget)

    # TODO: extract info from final recommendation instead of an auxiliary `throughputs` list
    throughputs = []
    for _ in range(0, parsed_args.search_budget,
                   parsed_args.num_compilation_processes):
      print(f'\n***********************************************')
      print(
          f'{search_strategy} optimization iter {_}/{parsed_args.search_budget}'
      )

      # 1. Launch `num_compilation_processes` compilation and "first-run" processes.
      ng_executions = []

      for compilation_number in range(parsed_args.num_compilation_processes):
        proposal = scheduler.optimizer.ask()

        # Create problem instance, which holds the compiled module and the
        # ExecutionEngine.
        problem_instance = ProblemInstance(problem_definition, [np.float32] * 3)

        # The optimizer may chose to ignore our constraints.
        # Override this, do not evaluate and give it max cost.
        if not scheduler.validate_proposal(proposal):
          ng_executions.append(
              NGMPExecution(proposal=proposal,
                            problem_instances=None,
                            process=None,
                            ipc_state=None))
          continue

        # Start process that compiles and runs.
        ipc_dict = manager.dict()
        p = mp.Process(target=compile_and_run_checked_mp,
                       args=[
                           problem_instance, scheduler, proposal,
                           parsed_args.n_iters, ipc_dict
                       ])
        # Best effort pin process in a round-robin fashion.
        os.system(
            f'taskset -p -c {compilation_number % os.cpu_count()} {p.pid}')
        p.start()
        # Append NGMPExecution. After synchronization results will be available
        # in ipc_dict['result'].
        ng_executions.append(
            NGMPExecution(proposal=proposal,
                          problem_instance=problem_instance,
                          process=p,
                          ipc_dict=ipc_dict))

      # 2. Join all processes.
      join_bulk_synchronous_processes(ng_executions,
                                      parsed_args.timeout_per_compilation)

      # 3. Inspect the ipc_state.
      # This is the result of a noisy run but it is cheap to evaluate.
      for ng_mp_execution in ng_executions:
        ipc_state = ng_mp_execution.future_ipc_state()

        if not ipc_state.success:
          scheduler.optimizer.tell(ng_mp_execution.proposal, 1e9)
          continue

        process_throughputs = ipc_state.throughputs[
            parsed_args.metric_to_measure]
        # Calculate the relative distance to peak: invert the throughput @75%
        # (i.e. 5th computed quantile).
        # Lower is better.
        throughput = compute_quantiles(process_throughputs)[5]
        relative_error = \
          (parsed_args.machine_peak - throughput) / parsed_args.machine_peak
        scheduler.optimizer.tell(ng_mp_execution.proposal, relative_error)
        # TODO: extract info from final recommendation instead of an auxiliary `throughputs` list
        throughputs.append(throughput)

    # TODO: better handling of result saving, aggregation etc etc.
    final_module_filename = None
    if parsed_args.output_dir is not None:
      final_module_filename = f'{parsed_args.output_dir}/module.mlir'
    else:
      final_module_filename = '/tmp/module.mlir'

    recommendation = scheduler.optimizer.recommend()
    # TODO: extract information from saved and draw some graphs
    # TODO: extract info from final recommendation instead of an auxiliary `throughputs` list
    throughputs.sort()
    best = int(throughputs[-1])
    print(
        f'Best solution: {best} GUnits/s (peak is {parsed_args.machine_peak} GUnits/s)'
    )
    scheduler.save_proposal_as_module(
        proposal=recommendation,
        module_save_filename=final_module_filename,
        benefit=best)


def main():
  print(f'Available optimizers\n{sorted(ng.optimizers.registry.keys())}')
  argparser = ArgumentParser()
  add_argparser_arguments(argparser, default_problem_sizes_list=[[16, 16, 16]])
  add_argparser_tuning_arguments(argparser)
  parsed_args = argparser.parse_args()
  bulk_synchronous_optim_loop(parsed_args=parsed_args)


if __name__ == '__main__':
  main()
