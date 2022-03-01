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


class SearchJobResult():
  """The result of a search job."""

  def __init__(self, proposal, throughputs):
    self.proposal = proposal
    self.throughputs = throughputs


class ProcessState():
  """State that is shared among all jobs that are scheduled to run on a pool
  process.
  """

  def __init__(self, available_cpus_queue, parsed_args, results_queue, rw_locks,
               shutdown_event):
    self.available_cpus_queue = available_cpus_queue
    self.parsed_args = parsed_args
    self.results_queue = results_queue
    self.rw_locks = rw_locks
    self.shutdown_event = shutdown_event


def cpu_count():
  """Return the number of CPUs on which processes can be scheduled."""
  return len(os.sched_getaffinity(0))

def compile_and_run_checked_mp(problem: ProblemInstance, \
                               proposal,
                               scheduler: NGSchedulerInterface):
  """Entry point to compile and run while catching and reporting exceptions.

  This is run in interruptible multiprocess mode. Results are returned from this
  process via `result_queue`.

  Before starting compilation, acquire a lock in read mode. Wait for a certain
  amount of time and kill the process if compilation is still running. The
  compilation is always run on a single CPU. (The process is already pinned to a
  single CPU when entering this function.)

  Before starting a benchmark, acquire a lock in write mode. Wait for a certain
  amount of time and kill the process if the benchmark is still running. The
  benchmark may be run on multiple CPUs.

  Upon SIGTERM triggered by the parent, locks are simply dropped on the floor
  and everything stops.
  """

  global process_state

  # During pool initialization, every process is pinned to one CPU for
  # compilation purposes. Based on this CPU ID, we can compute the benchmark
  # CPU range and determine which lock must be acquired.
  assert len(os.sched_getaffinity(0)) == 1, \
    "expected that process is pinnen to single CPU"
  compilation_cpu = os.sched_getaffinity(0).pop()
  num_cpus_per_benchmark = process_state.parsed_args.num_cpus_per_benchmark

  # There is one reader-write lock per [benchmark_cpu_min; benchmark_cpu_max)
  # CPU range. Every CPU range consists of `num_cpus_per_benchmark` consecutive
  # CPUs.
  lock_idx = int(compilation_cpu / num_cpus_per_benchmark)
  benchmark_cpu_min = lock_idx * num_cpus_per_benchmark
  benchmark_cpu_max = benchmark_cpu_min + num_cpus_per_benchmark
  lock = process_state.rw_locks[lock_idx]

  # Sanity check: ensure compilation and benchmark on the same set of CPUs.
  assert benchmark_cpu_min <= compilation_cpu and compilation_cpu < benchmark_cpu_max, \
    f'compilation_cpu: {compilation_cpu} is not in range [{benchmark_cpu_min}, {benchmark_cpu_max})'

  entry_point_name = 'main'
  fun_to_benchmark_name = 'fun_to_benchmark'

  # Function called in the compilation thread.
  def compile():
    # Construct the schedule and save the module in case we need to replay later.
    def schedule_and_save(module):
      scheduler.schedule(module, proposal)
      # TODO: save and report on error.

    problem.compile_with_schedule_builder( \
      entry_point_name=entry_point_name,
      fun_to_benchmark_name=fun_to_benchmark_name,
      compile_time_problem_sizes_dict=scheduler.
      build_compile_time_problem_sizes(),
      schedule_builder=schedule_and_save)

  throughputs_placeholder = []

  # Function called in the benchmark thread.
  def benchmark():
    throughputs_placeholder.append(
        problem.run(
            n_iters=process_state.parsed_args.n_iters,
            entry_point_name=entry_point_name,
            runtime_problem_sizes_dict=problem.compile_time_problem_sizes_dict))

  # Function called when compilation or benchmark times out.
  def kill_process():
    # Put the compilation CPU back into the queue, so that a newly spawned
    # process can pin itself to that CPU. (After this process was terminated.)
    process_state.available_cpus_queue.put(compilation_cpu)
    process_state.results_queue.put(SearchJobResult(proposal, None))
    exit(1)

  try:
    f = io.StringIO()
    with redirect_stdout(f):
      # Compilation: Process is already pinned to compliation_cpu.
      print(f'compilation_cpu: {compilation_cpu}')

      # Acquire a read lock for compilation. Multiple compilations may run in
      # parallel on this CPU range.
      lock.acquire_read()

      # Start compilation.
      t = threading.Thread(target=compile)
      t.start()

      # Kill process after compilation timeout.
      t.join(process_state.parsed_args.timeout_per_compilation)
      lock.release()
      if t.is_alive():
        kill_process()

      # Acquire a write lock for benchmarking. No other compilation or benchmark
      # may run on this CPU range.
      lock.acquire_write()

      # Benchmark: Pin process to the specified CPU range.
      os.sched_setaffinity(
          0, [x for x in range(benchmark_cpu_min, benchmark_cpu_max)])

      # Start benchmark.
      t = threading.Thread(target=benchmark)
      t.start()

      # Kill process after benchmark timeout.
      t.join(process_state.parsed_args.timeout_per_benchmark)
      lock.release()
      if t.is_alive():
        kill_process

      # Pin this process back to the compilation CPU.
      os.sched_setaffinity(0, {compilation_cpu})

    # TODO: redirect to a file if we want this information.
    f.flush()
    if len(throughputs_placeholder) == 0:
      process_state.results_queue.put(SearchJobResult(proposal, None))
    else:
      process_state.results_queue.put(
          SearchJobResult(proposal, throughputs_placeholder[0]))

  except Exception as e:
    traceback.print_exc()
    lock.release()
    # TODO: save to replay errors.
    print(e)


def tell_optimizer(
    optimizer,
    result,
    # TODO: extract info from final recommendation instead
    # of an auxiliary `throughputs` list.
    throughputs: tp.Sequence[float],
    parsed_args):
  """Tell the result for the proposal."""

  if not result.throughputs:
    optimizer.tell(result.proposal, 1)
    return 0

  process_throughputs = result.throughputs[parsed_args.metric_to_measure]
  # Calculate the relative distance to peak: invert the throughput @90%
  # (i.e. 6th computed quantile).
  # Lower is better.
  # This matches the optimization process which is a minimization.
  throughput = compute_quantiles(process_throughputs)[6]
  relative_error = \
    (parsed_args.machine_peak - throughput) / parsed_args.machine_peak
  optimizer.tell(result.proposal, relative_error)
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
### Multiprocess optimization loop.
################################################################################

def async_optim_loop(problem_definition: ProblemDefinition, \
                     scheduler: NGSchedulerInterface,
                     optimizer,
                     parsed_args):
  """Asynchronous NG scheduling with multi-process evaluation of proposals."""

  # Helper variables.
  num_cpus_per_benchmark = parsed_args.num_cpus_per_benchmark

  # A benchmark may run on multiple CPUs. Must divide cpu_count evenly.
  assert cpu_count() % parsed_args.num_cpus_per_benchmark == 0, \
    f'num_cpus_per_benchmark: {parsed_args.num_cpus_per_benchmark} ' + \
    f'must divide the number of cpus: {cpu_count()}'
  num_concurrent_benchmarks = int(cpu_count() / num_cpus_per_benchmark)

  # Reader-writer locks are used to ensure that nothing else in running on a
  # CPU range while a benchmark is running. If no benchmark is running, multiple
  # compilations may be running on a CPU range (even more than cores available).
  mp_manager = mp.Manager()
  benchmark_locks = [RWLock() for idx in range(num_concurrent_benchmarks)]

  # Concurrent queue for benchmark results.
  results_queue = mp_manager.Queue()

  # Concurrent queue for available CPUs. Every mp.Pool process is pinned to one
  # CPU from the queue. During the benchmark phase, it is pinned to a CPU range.
  available_cpus_queue = mp_manager.Queue()
  for i in range(parsed_args.num_parallel_tasks):
    available_cpus_queue.put(i % cpu_count())

  shutdown_event = mp_manager.Event()

  def init_proccess(process_state_arg):
    """Initialize an mp.Pool process.

    Store process_state in a global variable, so that it can be shared among all
    jobs that are run on this process and be accessed during job execution. Also
    select a pinned compilation CPU for this process.
    """

    # Store process_state in global variable.
    global process_state
    process_state = process_state_arg

    # Install signal handler to stop program execution on CTRL+C.
    def signal_handler(sig, frame):
      process_state.shutdown_event.set()
      exit(1)

    signal.signal(signal.SIGINT, signal_handler)

    # Pin this process to a single CPU (for compilation).
    cpu_id = process_state.available_cpus_queue.get()
    os.sched_setaffinity(0, {cpu_id})

  # Initialize the process pool. This also initializes all worker processes. If
  # a worker process is killed during execution, a new process is spawned and
  # initialized, so that there are always `num_parallel_tasks` many processes.
  process_state = ProcessState(available_cpus_queue, parsed_args, results_queue,
                               benchmark_locks, shutdown_event)
  process_pool = mp.Pool(processes=parsed_args.num_parallel_tasks,
                         initializer=init_proccess,
                         initargs=(process_state,))

  def enqueue_search_job():
    """Ask for the next proposal and schedule a run on the pool."""
    proposal = optimizer.ask()

    # Create problem instance, which holds the compiled module and the
    # ExecutionEngine.
    problem_types = [np.float32] * 3
    problem_instance = ProblemInstance(problem_definition, problem_types)

    # Enqueue the job that compiles and runs.
    process_pool.apply_async(func=compile_and_run_checked_mp,
                             args=(problem_instance, proposal, scheduler))

  # TODO: extract info from final recommendation instead of an auxiliary
  # `throughputs` list.
  search_number = 0
  throughputs = []

  def shutdown():
    print(f'Ctrl+C received from pid {os.getpid()}')
    try:
      process_pool.terminate()
    except:
      pass
    print('Killed children processes')
    finalize_parallel_search(scheduler, optimizer, throughputs, parsed_args)
    exit(1)

  def signal_handler(sig, frame):
    shutdown()

  signal.signal(signal.SIGINT, signal_handler)

  # Enqueue slightly more jobs than processes, so that another job can start
  # running immediately when a job finishes. When a job finishes, another job
  # is immediately enqueued.
  num_initial_jobs = min(parsed_args.search_budget,
                         int(1.5 * parsed_args.num_parallel_tasks))
  search_number = num_initial_jobs
  for _ in range(num_initial_jobs):
    enqueue_search_job()

  # Wait for the `search_budget` many results.
  best = 0
  num_failed = 0
  for i in range(parsed_args.search_budget):
    if i % 10 == 1:
      if shutdown_event.is_set():
        shutdown()

      sys.stdout.write(f'*******\t' +
                       f'{parsed_args.search_strategy} optimization iter ' +
                       f'{i} / {parsed_args.search_budget}, ' +
                       f'best so far: {int(best)} GUnits/s, ' +
                       f'#failed: {num_failed}\r')
      sys.stdout.flush()

    # Retrieve a result from the queue.
    result = results_queue.get()
    if not result.throughputs:
      num_failed += 1
    throughput = tell_optimizer(optimizer, result, throughputs, parsed_args)
    best = throughput if throughput > best else best

    if search_number < parsed_args.search_budget:
      search_number += 1
      enqueue_search_job()

  finalize_parallel_search(scheduler, optimizer, throughputs, parsed_args)

  print('Done')
