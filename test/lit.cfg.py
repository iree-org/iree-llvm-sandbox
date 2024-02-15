import os

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'Structured'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir', '.textpb', '.py']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.structured_build_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

config.substitutions.extend([
    ("%PYTHON", sys.executable),
])

# Deactivate Python's output buffering. Otherwise, output on stdout from Python
# gets delayed with respect to output from native libraries (such as the MLIR
# Python bindings) such that order is not preserved and FileCheck checks fail.
config.environment['PYTHONUNBUFFERED'] = '1'

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "lit.cfg.py",
    "lit.site.cfg.py",
]

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)


# Copied from `third_party/llvm-project/mlir/test/lit.cfg.py`.
# Searches for a runtime library with the given name and returns a tool
# substitution of the same name and the found path.
# Correctly handles the platforms shared library directory and naming conventions.
def add_runtime(name):
  path = ""
  for prefix in ["", "lib"]:
    path = os.path.join(config.llvm_shlib_dir,
                        f"{prefix}{name}{config.llvm_shlib_ext}")
    if os.path.isfile(path):
      break
  return ToolSubst(f"%{name}", path)


mlir_async_runtime = add_runtime("mlir_async_runtime")
mlir_c_runner_utils = add_runtime("mlir_c_runner_utils")
mlir_runner_utils = add_runtime("mlir_runner_utils")

config.environment['MLIR_ASYNC_RUNTIME_LIB'] = mlir_async_runtime.command
config.environment['MLIR_C_RUNNER_UTILS_LIB'] = mlir_c_runner_utils.command
config.environment['MLIR_RUNNER_UTILS_LIB'] = mlir_runner_utils.command

config.structured_tools_dir = os.path.join(config.structured_build_root, 'bin')
tool_dirs = [config.structured_tools_dir, config.llvm_tools_dir]
tools = [
    mlir_async_runtime,
    mlir_c_runner_utils,
    mlir_runner_utils,
    'structured-opt',
    ToolSubst('%mlir_lib_dir', config.mlir_lib_dir),
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# Pass through LLVM_SYMBOLIZER_PATH from environment
if "LLVM_SYMBOLIZER_PATH" in os.environ:
  config.environment["LLVM_SYMBOLIZER_PATH"] = \
      os.environ["LLVM_SYMBOLIZER_PATH"]

structured_python_path = os.path.join(config.structured_build_root,
                                      'python_packages')
llvm_config.with_environment('PYTHONPATH', [structured_python_path],
                             append_path=True)
