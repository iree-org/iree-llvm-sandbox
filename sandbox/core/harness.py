import time

from collections.abc import Callable


# Log everything to stderr and flush so that we have a unified stream to match
# errors/info emitted by MLIR to stderr.
def log(*args):
  print(*args, file=sys.stderr)
  sys.stderr.flush()


def timed_invoke(execute: Callable, gflop_count: int, n_iters: int, args: list,
                 **kwargs):
  start = time.time()
  execute(*args, **kwargs, n_iters=n_iters)
  elapsed_s = time.time() - start
  elapsed_s_per_iter = elapsed_s / n_iters
  gflop_per_s_per_iter = gflop_count / (elapsed_s_per_iter)
  print(f'xxxxxxxxxx : {n_iters} iters time on {1} threads '
        f'in {elapsed_s_per_iter:.{4}}s per iter '
        f'sec ({gflop_per_s_per_iter:.{4}} GFlop/s) '
        f'total time {elapsed_s:.{4}}s ')


def setup_and_invoke(
    setup_fun: Callable,
    run_fun: Callable,
    n_iters: int,
    gflop_count: int,
    n_iters_dry_run=1,
    # Used in particular to pass a check_fun callable.
    **kwargs):
  tensors = setup_fun()

  module, execution_engine = None, None
  if 'compile_fun' in kwargs:
    module, execution_engine = kwargs['compile_fun'](*tensors)

  # Dry-run.
  if n_iters_dry_run > 0:
    timed_invoke(
        run_fun,
        gflop_count,
        n_iters_dry_run,
        tensors,
        module=module,
        execution_engine=execution_engine)

  if 'check_fun' in kwargs:
    kwargs['check_fun'](*tensors)

  # Run for ITERS and report timing.
  timed_invoke(
      run_fun,
      gflop_count,
      n_iters,
      tensors,
      module=module,
      execution_engine=execution_engine)
