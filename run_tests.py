#!/usr/bin/env python
# Script to run tests and benchmarks.

import glob
import os
import subprocess
import sys


def _convert_path_to_module(test_script : str) -> str:
  """Convert the path of the test script to its module name."""
  test_script = test_script.replace(os.sep, ".")
  test_script = test_script.strip(".")
  if test_script.endswith(".py"):
    return test_script[:-3]
  return test_script


def _run_test(test_script: str) -> bool:
  """Run the provided test script an return failure or success.

  A test succeeds if:
  - it does not time out
  - it returns zero
  - it does not print FAILURE
  """
  print(f"- running {test_script}: ", end="")
  module = _convert_path_to_module(test_script)
  environment = os.environ
  environment["PYTHONPATH"] = "./build/tools/sandbox/python_package"
  environment["MLIR_RUNNER_UTILS_LIB"] = "./build/lib/libmlir_runner_utils.so"
  proc = subprocess.Popen(["python", "-m", module],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          env=environment)
  try:
    outs, errs = proc.communicate(timeout=20)
  except subprocess.TimeoutExpired:
    proc.kill()
    print("\033[31m" + "FAILED" + "\033[m")
    print("  -> test execution timed out")
    return False
  if proc.returncode != 0:
    print("\033[31m" + "FAILED" + "\033[m")
    print(f"  -> test returned code {proc.returncode}")
    return False
  # Check the output for numerical failures.
  outs = outs.decode("utf-8")
  errs = errs.decode("utf-8")
  for line in outs.splitlines() + errs.splitlines():
    if line.count("FAILURE") != 0:
      print("\033[31m" + "FAILED" + "\033[m")
      print(f"  -> test failure: {line}")
      return False
  print("\033[32m" + "SUCCESS" + "\033[m")
  return True


def main():
  results = []
  for f in glob.glob("./python/**/*test.py", recursive=True):
    results.append(_run_test(f))
  errors = results.count(False)
  if errors:
    print(f'-> {errors} tests failed!')
    exit(1)

if __name__ == '__main__':
  main()
