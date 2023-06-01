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
config.suffixes = ['.mlir', '.py']

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

config.structured_tools_dir = os.path.join(config.structured_build_root, 'bin')
tool_dirs = [config.structured_tools_dir, config.llvm_tools_dir]
tools = [
    'structured-opt',
    ToolSubst('%mlir_lib_dir', config.mlir_lib_dir),
]

# Pass through LLVM_SYMBOLIZER_PATH from environment
if "LLVM_SYMBOLIZER_PATH" in os.environ:
  config.environment["LLVM_SYMBOLIZER_PATH"] = \
      os.environ["LLVM_SYMBOLIZER_PATH"]


def add_runtime(name):
  for prefix in ['', 'lib']:
    path = os.path.join(config.mlir_lib_dir,
                        f'{prefix}{name}{config.llvm_shlib_ext}')
    if os.path.isfile(path):
      break
  return path


config.environment["MLIR_RUNNER_UTILS_LIB"] = add_runtime('mlir_runner_utils')
config.environment["MLIR_C_RUNNER_UTILS_LIB"] = add_runtime(
    'mlir_c_runner_utils')
config.environment["MLIR_ASYNC_RUNTIME_LIB"] = add_runtime('mlir_async_runtime')

llvm_config.add_tool_substitutions(tools, tool_dirs)

structured_python_path = os.path.join(config.structured_build_root,
                                      'python_packages')
llvm_config.with_environment('PYTHONPATH', [structured_python_path],
                             append_path=True)
