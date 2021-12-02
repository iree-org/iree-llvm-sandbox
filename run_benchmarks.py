#!/usr/bin/env python
# Script to run tests and benchmarks.

import io
import importlib
import glob
import os
import subprocess
import sys

from contextlib import redirect_stdout

def _convert_path_to_module(test_script : str) -> str:
  """Convert the path of the test script to its module name."""
  test_script = test_script.replace(os.sep, ".")
  test_script = test_script.strip(".")
  if test_script.endswith(".py"):
    return test_script[:-3]
  return test_script


def _run_benchmark(bench_script: str):
  """Check if the provided script has a benchmark() function and run it."""
  module_name = _convert_path_to_module(bench_script)
  module = importlib.import_module(module_name)
  if not hasattr(module, "benchmark"):
    return
  print(f"- running {bench_script}:")
  f = io.StringIO()
  with redirect_stdout(f):
    module.benchmark()
  out = f.getvalue()
  expert = 0
  for line in out.splitlines():
    if "Runtime problem size" in line:
      sizes = [x.split(",")[0].split("}")[0] for x in line.split(":")[1:]]
      problem_size = "x".join([x.strip() for x in sizes])
      print(f"  -> problem {problem_size}:\t")
      expert = 0
    if "expert" in line:
      print(f"     - expert[{expert}]:\t\t", end="")
      expert = expert + 1
    if "GFlops/s" in line:
      compute_throughputs = line.split()
      print(f"{compute_throughputs[-2]} GFlops/s\t\t", end="")
    if "GBs/s" in line:
      memory_throughputs = line.split()
      print(f"{memory_throughputs[-2]} GBs/s")


def main():
  for f in glob.glob("./python/**/*bench.py", recursive=True):
    _run_benchmark(f)


if __name__ == '__main__':
  main()
