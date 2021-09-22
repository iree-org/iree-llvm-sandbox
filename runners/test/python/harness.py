import time

from collections.abc import Callable

# Log everything to stderr and flush so that we have a unified stream to match
# errors/info emitted by MLIR to stderr.
def log(*args):
  print(*args, file=sys.stderr)
  sys.stderr.flush()


def timed_invoke(execute: Callable, gflop_count: int, n_iters: int, args: list):
  start = time.time()
  execute(n_iters, *args)
  elapsed_s_per_iteration = (time.time() - start) / n_iters
  gflop_per_s_per_iteration = gflop_count / (elapsed_s_per_iteration)
  return elapsed_s_per_iteration, gflop_per_s_per_iteration


def setup_and_invoke(
    setup_fun: Callable,
    run_fun: Callable,
    n_iters: int,
    gflop_count: int,
    n_iters_dry_run=1,
    # Used in particular to pass a check_fun callable.
    **kwargs):
  tensors = setup_fun()

  # Dry-run.
  if n_iters_dry_run > 0:
    elapsed_s_per_iter, gflop_per_s_per_iter = \
        timed_invoke(run_fun, gflop_count, n_iters_dry_run, tensors)
    print(f'dry_run in {elapsed_s_per_iter:.{4}}s per iter '
          f'sec ({gflop_per_s_per_iter:.{4}} GFlop/s) ')

  # Run for ITERS and report timing.
  elapsed_s_per_iter, gflop_per_s_per_iter = \
      timed_invoke(run_fun, gflop_count, n_iters, tensors)
  print(f'xxxxxxxxxx : time on {1} threads '
        f'in {elapsed_s_per_iter:.{4}}s per iter '
        f'sec ({gflop_per_s_per_iter:.{4}} GFlop/s) ')

  if kwargs['check_fun']:
    kwargs['check_fun'](*tensors)
