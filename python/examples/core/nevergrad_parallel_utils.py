from contextlib import redirect_stdout, redirect_stderr
import io
import math
import multiprocessing as mp
import numpy as np
import os
from prwlock import RWLock
import signal
import sys
import threading
import time
import traceback
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

  def __init__(self, proposal, problem_instance, process, ipc_dict):
    super().__init__(proposal, problem_instance)
    self.process = process
    self.ipc_dict = ipc_dict
    self.joined_with_root = False

  def ipc_state(self):
    return self.ipc_dict['result'] if 'result' in self.ipc_dict else \
      IPCState(success=False, throughputs=None)


class ProcessManagementHelper(object):
  """Helper class to centralize the management of task and lock lists.

  The general setup is that every compile + benchmark job is mapped to the first  
  free slot in the `tasks` list; i.e. in the range:
    0 1 2 .... parsed_args.num_parallel_tasks

  The compilation phase of a task "idx" is pinned to CPU "idx % cpu_count(). 
  If num_parallel_tasks > cpu_count, multiple compilation phases may be pinned 
  to the same CPU and run in parallel. 
  Before the compilation begins, a reader lock is acquired for the pinned CPU.

  The benchmark phase of a task "idx" is pinned to the CPU range 
  "[floor((idx % cpu_count()) / num_cpus_per_benchmark) * num_cpus_per_benchmark; 
   (floor((idx % cpu_count()) / num_cpus_per_benchmark) + 1) * num_cpus_per_benchmark)", 
  which exclusively runs the benchmark of task "idx". 
  Before the benchmark starts, a writer lock is acquired.

  Note that all all CPUs in a benchmark CPU range share the same lock.
  As a consequence, each lock `lock_idx` locks CPUs in the range: 
    [lock_idx * parsed_args.num_cpus_per_benchmark, 
    (lock_idx + 1) * parsed_args.num_cpus_per_benchmark - 1).
  """

  def __init__(self, num_cpus_per_benchmark: int, num_parallel_tasks: int):
    self.num_cpus_per_benchmark = num_cpus_per_benchmark
    assert self.cpu_count() % num_cpus_per_benchmark == 0, \
      f'num_cpus_per_benchmark: {num_cpus_per_benchmark} ' + \
      f'must divide the number of cpus: {self.cpu_count()}'

    self.num_concurrent_benchmarks = \
      int(self.cpu_count() / self.num_cpus_per_benchmark)

    self.mp_manager = mp.Manager()
    self.num_parallel_tasks = num_parallel_tasks
    self.tasks = [None] * self.num_parallel_tasks
    self.benchmark_locks = [
        RWLock() for idx in range(self.num_concurrent_benchmarks)
    ]

  def get_benchmark_lock_idx(self, task_idx: int):
    """Get the index of the lock to run the benchmarking of task task_idx.
    """
    num_tasks_per_lock = math.ceil((task_idx + 1) / len(self.benchmark_locks))
    return int(task_idx / num_tasks_per_lock)

  def get_benchmark_lock(self, task_idx: int):
    """Get the lock to run the benchmarking of task task_idx.
    """
    return self.benchmark_locks[self.get_benchmark_lock_idx(task_idx)]

  def cpu_count(self):
    """Return the number of CPUs on which processes can be scheduled."""
    return len(os.sched_getaffinity(0))

  def get_compilation_cpu(self, task_idx: int):
    """Get the (single) CPU to run the compilation of task task_idx.
    """
    cpu_min, cpu_max = self.get_benchmark_cpu_range(task_idx)
    return cpu_min + task_idx % (cpu_max - cpu_min)

  def get_benchmark_cpu_range(self, task_idx: int):
    """ Get the [min, max) CPU range to run the benchmarking of task task_idx.
    """
    lock_idx = self.get_benchmark_lock_idx(task_idx)
    return lock_idx * self.num_cpus_per_benchmark, \
           (lock_idx + 1) * self.num_cpus_per_benchmark

  def get_first_free_slot_or_sync(self, scheduler, optimizer, throughputs,
                                  parsed_args):
    """Return the first index of a None entry in self.tasks.
    
    Wait and join at least one process if no such entry exists.
    """
    best = 0
    # Find the first empty slot in tasks.
    if not None in self.tasks:
      # Join at least one process if nothing is available.
      processes_joined = join_at_least_one_process(self.tasks)
      assert len(processes_joined) > 0, "no processes were joined"
      for process_idx in processes_joined:
        throughput = tell_joined_process(self.tasks, process_idx, scheduler,
                                         optimizer, throughputs, parsed_args)
        throughput = int(throughput)
        best = throughput if throughput > best else best
        self.tasks[process_idx] = None

    # We are sure there is at least one empty slot.
    return self.tasks.index(None), best


def compile_and_run_checked_mp(problem: ProblemInstance, \
                               lock: RWLock,
                               compilation_cpu: int,
                               benchmark_cpu_min: int,
                               benchmark_cpu_max: int,
                               scheduler: NGSchedulerInterface,
                               proposal,
                               n_iters: int,
                               timeout_compile: float,
                               timeout_benchmark: float,
                               ipc_dict: dict):
  """Entry point to compile and run while catching and reporting exceptions.

  This is run in interruptible multiprocess mode.

  ipc_dict must be provided, and it is used to return information across the
  root / children process boundary:
    - 'throughputs': the measured throughputs.
    - 'success': the return status.

  Acquire lock in read mode and spawn a compile thread, joined with a 
  `timeout_compile` timeout to perform the compile part of the job. 
  The compilation is always run on a single CPU, set by a taskset command.

  Acquire lock in write mode and spawn a benchmark thread, joined with a 
  `timeout_benchmark` timeout to perform the benchmark part of the job. 
  The benchmark may be run on multiple CPUs, set by a taskset command.

  Upon SIGTERM triggered by the parent, locks are simply dropped on the floor
  and everything stops.
  """
  # Sanity check: ensure compilation and benchmark on the same set of CPUs.
  assert benchmark_cpu_min <= compilation_cpu and compilation_cpu < benchmark_cpu_max, \
    f'compilation_cpu: {compilation_cpu} is not in range [{benchmark_cpu_min}, {benchmark_cpu_max})'

  # Function called in the compilation thread.
  def compile():
    # Construct the schedule and save the module in case we need to replay later.
    def schedule_and_save(module):
      scheduler.schedule(module, proposal)
      # TODO: save and report on error.

    problem.compile_with_schedule_builder( \
      entry_point_name=scheduler.entry_point_name,
      fun_to_benchmark_name=scheduler.fun_to_benchmark_name,
      compile_time_problem_sizes_dict=scheduler.
      build_compile_time_problem_sizes(),
      schedule_builder=schedule_and_save)

  throughputs_placeholder = []

  # Function called in the benchmark thread.
  def benchmark():
    throughputs_placeholder.append(
        problem.run(
            n_iters=n_iters,
            entry_point_name=scheduler.entry_point_name,
            runtime_problem_sizes_dict=problem.compile_time_problem_sizes_dict))

  try:

    f = io.StringIO()
    with redirect_stdout(f):
      # Pin compile part of the process to the unique compilation_cpu.
      print(f'compilation_cpu: {compilation_cpu}')
      os.sched_setaffinity(os.getpid(), [compilation_cpu])

      # Acquire, spawn a new thread, start and join with timeout, release.
      lock.acquire_read()
      t = threading.Thread(target=compile)
      t.start()
      t.join(timeout_compile)
      lock.release()
      if t.is_alive():
        exit(1)

      # Pin benchmark part of the process to the cpu_min-cpu_max range to allow
      # parallel benchmarks without prohibitive interferences.
      os.sched_setaffinity(
          os.getpid(), [x for x in range(benchmark_cpu_min, benchmark_cpu_max)])
      # Acquire, spawn a new thread, start and join with timeout, release.
      lock.acquire_write()
      t = threading.Thread(target=benchmark)
      t.start()
      t.join(timeout_benchmark)
      lock.release()
      if t.is_alive():
        exit(1)

    # TODO: redirect to a file if we want this information.
    f.flush()

    ipc_dict['result'] = IPCState(success=True,
                                  throughputs=throughputs_placeholder[0])

  except Exception as e:
    traceback.print_exc()
    lock.release()
    # TODO: save to replay errors.
    print(e)

def ask_and_fork_process(pmh: ProcessManagementHelper, \
                         problem_definition: ProblemDefinition,
                         problem_types: tp.Sequence[np.dtype],
                         task_idx: int,
                         scheduler: NGSchedulerInterface,
                         optimizer,
                         parsed_args):
  """Ask for the next proposal and fork its evaluation in a new process"""

  proposal = optimizer.ask()

  # Create problem instance, which holds the compiled module and the
  # ExecutionEngine.
  problem_instance = ProblemInstance(problem_definition, problem_types)

  compilation_cpu = pmh.get_compilation_cpu(task_idx)
  benchmark_cpu_min, benchmark_cpu_max = pmh.get_benchmark_cpu_range(task_idx)
  lock = pmh.get_benchmark_lock(task_idx)

  # Start process that compiles and runs.
  ipc_dict = pmh.mp_manager.dict()
  p = mp.Process(target=compile_and_run_checked_mp,
                 args=[
                   problem_instance, \
                   lock,
                   compilation_cpu,
                   benchmark_cpu_min,
                   benchmark_cpu_max,
                   scheduler,
                   proposal,
                   parsed_args.n_iters,
                   parsed_args.timeout_per_compilation,
                   parsed_args.timeout_per_benchmark,
                   ipc_dict
                 ])
  p.start()

  # Append NGMPEvaluation. After synchronization results will be available
  # in ipc_dict['result'].
  pmh.tasks[task_idx] = NGMPEvaluation(proposal=proposal,
                                       problem_instance=problem_instance,
                                       process=p,
                                       ipc_dict=ipc_dict)


def tell_joined_process(tasks: tp.Sequence[NGMPEvaluation], \
                        task_idx: int,
                        scheduler: NGSchedulerInterface,
                        optimizer,
                        # TODO: extract info from final recommendation instead
                        # of an auxiliary `throughputs` list.
                        throughputs: tp.Sequence[float],
                        parsed_args):
  """Tell the result for the proposal from a joined evaluation process."""

  task = tasks[task_idx]
  ipc_state = task.ipc_state()

  if not ipc_state.success:
    optimizer.tell(task.proposal, 1)
    return 0

  process_throughputs = ipc_state.throughputs[parsed_args.metric_to_measure]
  # Calculate the relative distance to peak: invert the throughput @90%
  # (i.e. 6th computed quantile).
  # Lower is better.
  # This matches the optimization process which is a minimization.
  throughput = compute_quantiles(process_throughputs)[6]
  relative_error = \
    (parsed_args.machine_peak - throughput) / parsed_args.machine_peak
  optimizer.tell(task.proposal, relative_error)
  throughputs.append(throughput)
  return throughput


def finalize_parallel_search(scheduler: NGSchedulerInterface, \
                             optimizer,
                             throughputs: tp.Sequence[float],
                             parsed_args):
  """Report and save the best proposal after search finished."""
  if len(throughputs) == 0:
    return

  # TODO: better handling of result saving, aggregation etc etc.
  final_module_filename = None
  if parsed_args.output_dir is not None:
    final_module_filename = f'{parsed_args.output_dir}/module.mlir'
  else:
    final_module_filename = '/tmp/module.mlir'

  recommendation = optimizer.recommend()
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


def join_at_least_one_process(tasks: tp.Sequence[NGMPEvaluation]):
  """ Join at least one process in `tasks`.

  Note: `tasks` may contain `None` entries (i.e. empty slots).
  These are just skipped.

  The parent process performs busy-waiting until it has synchronized at least 
  one process (either because it terminated or because it timed out).
  """

  sleep_time = 1e-3

  # Iterate on tasks until we find one that has been marked 'joined'.
  done = False
  while not done:
    time.sleep(sleep_time)
    for idx in range(len(tasks)):
      # Invariant: Always decrease the timer for this process.
      task = tasks[idx]
      if task is None:
        continue

      process = task.process
      # This process finished by itself, join with it and mark it joined.
      if not process.is_alive():
        process.join()
        task.joined_with_root = True
        done = True
        continue

      # This process needs to continue.

    if done:
      break

  return [
      idx for idx, e in enumerate(tasks)
      if e is not None and e.joined_with_root == True
  ]


################################################################################
### Multiprocess optimization loops.
################################################################################

def async_optim_loop(problem_definition: ProblemDefinition, \
                     scheduler: NGSchedulerInterface,
                     optimizer,
                     parsed_args):
  """Asynchronous NG scheduling problem with multi-process evaluation of proposals.
  """

  # TODO: extract info from final recommendation instead of an auxiliary
  # `throughputs` list.
  search_number = 0
  throughputs = []
  pmh = ProcessManagementHelper(parsed_args.num_cpus_per_benchmark,
                                parsed_args.num_parallel_tasks)

  interrupted = [False]

  root_pid = os.getpid()

  def signal_handler(sig, frame):
    if root_pid != os.getpid():
      return
    interrupted[0] = True
    print(f'Ctrl+C received from pid {os.getpid()}')
    for idx, task in enumerate(pmh.tasks):
      if task is not None:
        task.process.kill()
        pmh.tasks[idx] = None
    print('Killed children processes')
    finalize_parallel_search(scheduler, optimizer, throughputs, parsed_args)
    exit(1)

  signal.signal(signal.SIGINT, signal_handler)

  best = 0
  while not interrupted[0] and search_number < parsed_args.search_budget:
    if search_number % 10 == 1:
      sys.stdout.write(f'*******\t' +
                       f'{parsed_args.search_strategy} optimization iter ' +
                       f'{search_number} / {parsed_args.search_budget}\t' +
                       f'best so far: {best} GUnits/s\r')
      sys.stdout.flush()

    # task_idx is an empty slot, subject to synchronizations.
    task_idx, best_synced = pmh.get_first_free_slot_or_sync(
        scheduler, optimizer, throughputs, parsed_args)
    best = best_synced if best_synced > best else best

    # Fill that empty slot.
    ask_and_fork_process(pmh, problem_definition, [np.float32] * 3, task_idx,
                         scheduler, optimizer, parsed_args)

    search_number = search_number + 1

  # Tell tail what's what.
  while not interrupted[0] and any(e is not None for e in pmh.tasks):
    # Join at least one process if nothing is available.
    tasks_joined = join_at_least_one_process(pmh.tasks)
    assert len(tasks_joined) > 0, "no processes were joined"
    for task_idx in tasks_joined:
      throughput = tell_joined_process(pmh.tasks, task_idx, scheduler,
                                       optimizer, throughputs, parsed_args)
      throughput = int(throughput)
      best = throughput if throughput > best else best
      pmh.tasks[task_idx] = None

  finalize_parallel_search(scheduler, optimizer, throughputs, parsed_args)

  print('Done')