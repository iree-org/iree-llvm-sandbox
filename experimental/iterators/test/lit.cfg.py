import os

import lit.formats
import lit.util

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'Iterators'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir', '.py']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.iterators_build_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))

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

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.iterators_build_root, 'test')
config.iterators_tools_dir = os.path.join(config.iterators_build_root, 'bin')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [config.iterators_tools_dir, config.llvm_tools_dir]
tools = [
    'iterators-opt',
]

# Pass through LLVM_SYMBOLIZER_PATH from environment
if "LLVM_SYMBOLIZER_PATH" in os.environ:
  config.environment["LLVM_SYMBOLIZER_PATH"] = \
      os.environ["LLVM_SYMBOLIZER_PATH"]

llvm_config.add_tool_substitutions(tools, tool_dirs)

itertors_python_path = os.path.join(config.iterators_build_root,
                                    'python_packages')
llvm_config.with_environment('PYTHONPATH', [itertors_python_path],
                             append_path=True)
