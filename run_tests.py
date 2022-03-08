#!/usr/bin/env python
# Script to run tests and benchmarks.
import argparse
import glob
import os
import subprocess
import sys

def parse_arguments():
  parser = argparse.ArgumentParser(description="Select tests to run")
  parser.add_argument(
      "--gpu-integration-tests",
      help="Run GPU integration tests - requires a GPU with CUDA installation.",
      dest="gpu_integration_tests",
      default=False,
      action=argparse.BooleanOptionalAction,
  )
  return parser.parse_args()


def _convert_path_to_module(test_script: str) -> str:
  """Convert the path of the test script to its module name."""
  test_script = test_script.replace(os.sep, ".")
  test_script = test_script.strip(".")
  if test_script.endswith(".py"):
    return test_script[:-3]
  return test_script


def _configure_env():
  env = os.environ
  build_dir = env["IREE_LLVM_SANDBOX_BUILD_DIR"]
  env["PYTHONPATH"] = (
      os.path.join(build_dir, "tools/sandbox/python_package") +
      ((":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else ""))
  env["MLIR_RUNNER_UTILS_LIB"] = os.path.join(build_dir,
                                              "lib/libmlir_runner_utils.so")
  env["MLIR_C_RUNNER_UTILS_LIB"] = os.path.join(
      build_dir, "lib/libmlir_c_runner_utils.so")
  env["MLIR_RUNNER_EXTRA_LIBS"] = os.path.join(
      build_dir, "lib/libmlir_async_runtime_copy.so")
  return env


def _run_test(test_script: str) -> bool:
  """Run the provided test script an return failure or success.
  A test succeeds if:
  - it does not time out
  - it returns zero
  - it does not print FAILURE
  """
  print(f"- running {test_script}: ", end="")
  module = _convert_path_to_module(test_script)
  env = _configure_env()
  proc = subprocess.Popen(["python", "-m", module],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          env=env)
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
    print(errs.decode("utf-8"))
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


def main(args):
  results = []
  for f in glob.glob("./python/**/*test.py", recursive=True):
    results.append(_run_test(f))
  errors = results.count(False)
  if errors:
    print(f"-> {errors} tests failed!")
  # Additionally run the lit tests.
  print(f"- running lit tests:")
  lit_args = ["lit", "-v", "test"]
  if not args.gpu_integration_tests:
    lit_args.append("--filter-out=Integration/Dialect/VectorExt/GPU")
  returncode = subprocess.call(lit_args, env=_configure_env())
  if returncode != 0:
    print(f"-> lit tests failed!")
  if returncode != 0 or errors:
    exit(1)


if __name__ == '__main__':
  main(parse_arguments())
