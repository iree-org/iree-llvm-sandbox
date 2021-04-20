import time

from collections.abc import Callable


# Log everything to stderr and flush so that we have a unified stream to match
# errors/info emitted by MLIR to stderr.
def log(*args):
  print(*args, file=sys.stderr)
  sys.stderr.flush()


# Counts FMA as 2 ops.
def gflop_count_matmul(M: int, N: int, K: int):
  return (2.0 * M * N * K) / 1e9


def timed_invoke(execute: Callable, n_iters: int, *args: list):
  start = time.time()
  execute(*args)
  elapsed_s_per_iteration = (time.time() - start) / n_iters
  gflop_per_s_per_iteration = gflop_count_matmul(*args[:3]) / (
      elapsed_s_per_iteration)
  return elapsed_s_per_iteration, gflop_per_s_per_iteration
