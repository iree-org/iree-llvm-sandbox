from contextlib import redirect_stdout, redirect_stderr
import io
import numpy as np
import multiprocessing as mp
import os
import signal
import sys
import typing as tp

from ..core.harness import *
from ..core.nevergrad_tuner_utils import NGSchedulerInterface
from ..core.problem_definition import ProblemDefinition
from ..core.utils import compute_quantiles


class IPCState:
  """State shared across parent and children processes in a MP run.
  
  All contained information must "pickle" (i.e. serialize across processes).
  """

  def __init__(self,
               success: bool,
               throughputs: tp.Sequence[float],
               problem=None):
    self.success = success
    self.throughputs = throughputs
    self.problem = problem


class NGEvaluation:
  """Handle to a nevergrad evaluation."""

  def __init__(self, proposal, problem_instance):
    self.proposal = proposal
    self.problem_instance = problem_instance


class NGMPEvaluation(NGEvaluation):
  """Handle to a multiprocess nevergrad evaluation."""

  def __init__(self, proposal, problem_instance, process, ipc_dict, time_left):
    super().__init__(proposal, problem_instance)
    self.process = process
    self.ipc_dict = ipc_dict
    self.time_left = time_left
    self.joined_with_root = False

  def ipc_state(self):
    return self.ipc_dict['result'] if 'result' in self.ipc_dict else None


def compile_and_run_checked_mp(problem: ProblemInstance, \
                               scheduler: NGSchedulerInterface,
                               proposal,
                               n_iters: int,
                               ipc_dict: dict):
  """Entry point to compile and run while catching and reporting exceptions.

  This can run in interruptible multiprocess mode.
  ipc_dict must be provided, and it is used to return information across the
  root / children process boundary:
    - 'throughputs': the measured throughputs.
    - 'success': the return status.
  """
  try:

    # Construct the schedule and save the module in case we need to replay later.
    def schedule_and_save(module):
      scheduler.schedule(module, proposal)
      # TODO: save and report on error.

    f = io.StringIO()
    with redirect_stdout(f):
      problem.compile_with_schedule_builder(
          entry_point_name=scheduler.entry_point_name,
          fun_to_benchmark_name=scheduler.fun_to_benchmark_name,
          compile_time_problem_sizes_dict= \
            scheduler.build_compile_time_problem_sizes(),
          schedule_builder=schedule_and_save)

      throughputs = problem.run(
          n_iters=n_iters,
          entry_point_name=scheduler.entry_point_name,
          runtime_problem_sizes_dict=problem.compile_time_problem_sizes_dict)

    # TODO: redirect to a file if we want this information.
    f.flush()

    ipc_dict['result'] = IPCState(success=True, throughputs=throughputs)
  except Exception as e:
    import traceback
    traceback.print_exc()
    # TODO: save to replay errors.
    print(e)
    ipc_dict['result'] = IPCState(success=False, throughputs=None)


def cpu_count():
  return len(os.sched_getaffinity(0))

def ask_and_fork_process(mp_manager: mp.Manager, \
                         problem_definition: ProblemDefinition,
                         problem_types: tp.Sequence[np.dtype],
                         ng_mp_evaluations: tp.Sequence[NGMPEvaluation],
                         evaluation_slot_idx: int,
                         scheduler: NGSchedulerInterface,
                         parsed_args):
  """Ask for the next proposal and fork its evaluation in a new process"""

  proposal = scheduler.optimizer.ask()

  # Create problem instance, which holds the compiled module and the
  # ExecutionEngine.
  problem_instance = ProblemInstance(problem_definition, problem_types)

  # The optimizer may chose to ignore our constraints.
  # Override this, do not evaluate and just give it max cost.
  if not scheduler.validate_proposal(proposal):
    ng_mp_evaluations[evaluation_slot_idx] = NGMPEvaluation(
        proposal=proposal,
        problem_instances=None,
        process=None,
        ipc_state=None,
        time_left=parsed_args.timeout_per_compilation)
    return

  # Start process that compiles and runs.
  ipc_dict = mp_manager.dict()
  p = mp.Process(target=compile_and_run_checked_mp,
                 args=[
                     problem_instance, scheduler, proposal, parsed_args.n_iters,
                     ipc_dict
                 ])
  p.start()
  # Best effort pin process in a round-robin fashion.
  # This is noisy so suppress its name.
  f = io.StringIO()
  with redirect_stdout(f):
    os.system(
        f'taskset -p -c {evaluation_slot_idx % cpu_count()} {p.pid} > /dev/null 2>&1'
    )
  # Append NGMPEvaluation. After synchronization results will be available
  # in ipc_dict['result'].
  ng_mp_evaluations[evaluation_slot_idx] = NGMPEvaluation(
      proposal=proposal,
      problem_instance=problem_instance,
      process=p,
      ipc_dict=ipc_dict,
      time_left=parsed_args.timeout_per_compilation)


def tell_joined_process(ng_mp_evaluations: tp.Sequence[NGMPEvaluation], \
                        evaluation_slot_idx: int,
                        scheduler: NGSchedulerInterface,
                        # TODO: extract info from final recommendation instead
                        # of an auxiliary `throughputs` list.
                        throughputs: tp.Sequence[float],
                        parsed_args):
  """Tell the result for the proposal from a joined evaluation process."""

  ng_mp_evaluation = ng_mp_evaluations[evaluation_slot_idx]
  ipc_state = ng_mp_evaluation.ipc_state()

  if not ipc_state.success:
    scheduler.optimizer.tell(ng_mp_evaluation.proposal, 1)
    return 0

  process_throughputs = ipc_state.throughputs[parsed_args.metric_to_measure]
  # Calculate the relative distance to peak: invert the throughput @90%
  # (i.e. 6th computed quantile).
  # Lower is better.
  # This matches the optimization process which is a minimization.
  throughput = compute_quantiles(process_throughputs)[6]
  relative_error = \
    (parsed_args.machine_peak - throughput) / parsed_args.machine_peak
  scheduler.optimizer.tell(ng_mp_evaluation.proposal, relative_error)
  throughputs.append(throughput)
  return throughput


def finalize_parallel_search(scheduler: NGSchedulerInterface, \
                             throughputs: tp.Sequence[float],
                             parsed_args):
  """Report and save the best proposal after search finished."""

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
  scheduler.save_proposal_as_module(proposal=recommendation,
                                    module_save_filename=final_module_filename,
                                    benefit=best)


################################################################################
### Multiprocess join support.
################################################################################


def join_at_least_one_process(ng_mp_evaluations: tp.Sequence[NGMPEvaluation]):
  """ Join at least one process in `ng_mp_evaluations`.

  Note: `ng_mp_evaluations` may contain `None` entries (i.e. empty slots).
  These are just skipped.

  The parent process performs busy-waiting until it has synchronized at least 
  one process (either because it terminated or because it timed out).
  
  Additionally, update `ng_mp_evaluations` for further processing across the 
  single / multi process boundary:
    1. on a timeout or an exception, the ipc_state entry in the corresponding
      ng_mp_evaluation has its success status set to False.
    2. on a successful finish, the ipc_state entry in the corresponding
      ng_mp_evaluations succeeded ad contains the throughputs. This is used to
      communicate back the compiled state of the problem across the single / 
      multiprocess boundary.
  """

  import time
  sleep_time = 0.1

  # Iterate on evaluations until we find one that has been marked 'joined'.
  done = False
  while not done:
    time.sleep(sleep_time)
    for idx in range(len(ng_mp_evaluations)):
      # Invariant: Always decrease the timer for this process.
      evaluation = ng_mp_evaluations[idx]
      if evaluation is None:
        continue
      evaluation.time_left = evaluation.time_left - sleep_time

      # This process was already joined, skip.
      evaluation = ng_mp_evaluations[idx]
      assert evaluation.joined_with_root == False, "Evaluation already joined"

      process = evaluation.process
      # This process finished by itself, join with it and mark it joined.
      if not process.is_alive():
        process.join()
        evaluation.joined_with_root = True
        done = True
        continue

      # This process timed out, terminate, join and mark it joined.
      if evaluation.time_left <= 0:
        f = io.StringIO()
        with redirect_stdout(f):
          print(f'timeout: {evaluation.proposal} did not complete')
          # TODO: redirect to a file if we want this information.
          f.flush()
        evaluation.process.terminate()
        evaluation.process.join()
        # Override to return a failed IPCState and signify infinite relative_error.
        evaluation.ipc_state = lambda: IPCState(success=False, throughputs=None)
        evaluation.joined_with_root = True
        done = True

      # This process needs to continue.

    if done:
      break

  return [
      idx for idx, e in enumerate(ng_mp_evaluations)
      if e is not None and e.joined_with_root == True
  ]


################################################################################
### Multiprocess optimization loops.
################################################################################

global interrupted

def async_optim_loop(problem_definition: ProblemDefinition, \
                     scheduler: NGSchedulerInterface,
                     parsed_args):
  """Asynchronous NG scheduling problem with multi-process evaluation of proposals.
  """
  mp_manager = mp.Manager()

  # TODO: extract info from final recommendation instead of an auxiliary `throughputs` list
  search_number = 0
  throughputs = []
  ng_mp_evaluations = [None] * parsed_args.num_compilation_processes

  interrupted = []

  def signal_handler(sig, frame):
    interrupted.append(True)

  signal.signal(signal.SIGINT, signal_handler)

  best = 0
  while len(interrupted) == 0 and search_number < parsed_args.search_budget:
    if search_number % 10 == 1:
      sys.stdout.write(f'*******\t' +
                       f'{parsed_args.search_strategy} optimization iter ' +
                       f'{search_number} / {parsed_args.search_budget}\t' +
                       f'best so far: {best} GUnits/s\r')
      sys.stdout.flush()

    # Find the first empty slot in ng_mp_evaluations.
    if not None in ng_mp_evaluations:
      # Join at least one process if nothing is available.
      processes_joined = join_at_least_one_process(ng_mp_evaluations)
      assert len(processes_joined) > 0, "no processes were joined"
      for process_idx in processes_joined:
        throughput = tell_joined_process(ng_mp_evaluations, process_idx,
                                         scheduler, throughputs, parsed_args)
        throughput = int(throughput)
        best = throughput if throughput > best else best
        ng_mp_evaluations[process_idx] = None

    # We are sure there is at least one empty slot.
    compilation_number = ng_mp_evaluations.index(None)

    # Fill that empty slot.
    ask_and_fork_process(mp_manager, problem_definition, [np.float32] * 3,
                         ng_mp_evaluations, compilation_number, scheduler,
                         parsed_args)

    search_number = search_number + 1

  if interrupted:
    for e in ng_mp_evaluations:
      if e is not None:
        e.time_left = 0
    print('\n')

  # Tell tail what's what.
  while any(e is not None for e in ng_mp_evaluations):
    # Join at least one process if nothing is available.
    processes_joined = join_at_least_one_process(ng_mp_evaluations)
    assert len(processes_joined) > 0, "no processes were joined"
    for process_idx in processes_joined:
      throughput = tell_joined_process(ng_mp_evaluations, process_idx,
                                       scheduler, throughputs, parsed_args)
      throughput = int(throughput)
      best = throughput if throughput > best else best
      ng_mp_evaluations[process_idx] = None

  finalize_parallel_search(scheduler, throughputs, parsed_args)
