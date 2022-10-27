import os
import sys

import lit.formats
import lit.util

import lit.llvm

# Configuration file for the 'lit' test runner.
lit.llvm.initialize(lit_config, config)

# name: The name of this test suite.
config.name = "Iterators MLIR"

config.test_format = lit.formats.ShTest(execute_external=True)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".py"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The path where tests shall execute.
build_dir = os.environ["IREE_LLVM_SANDBOX_BUILD_DIR"]
config.test_exec_root = os.path.join(build_dir, 'test')
lib_dir = os.path.join(build_dir, 'lib')

#config.use_default_substitutions()
config.excludes = [
    "lit.cfg.py",
    "lit.site.cfg.py",
]

config.substitutions.extend([
    ("%PYTHON", sys.executable),
])

# Pass through LLVM_SYMBOLIZER_PATH from environment
if "LLVM_SYMBOLIZER_PATH" in os.environ:
  config.environment["LLVM_SYMBOLIZER_PATH"] = \
      os.environ["LLVM_SYMBOLIZER_PATH"]

# Add the build/bin directory to the path.
sys.path.append(os.path.join(build_dir, "bin"))
config.environment["PYTHONPATH"] = ":".join(sys.path)
config.environment["PATH"] = ":".join(sys.path)
config.environment["RUNTIMELIB"] = os.path.join(lib_dir, 'libruntime_utils.so')
project_root = os.path.dirname(os.path.dirname(__file__))

# Deactivate Python's output buffering. Otherwise, output on stdout from Python
# gets delayed with respect to output from native libraries (such as the MLIR
# Python bindings) such that order is not preserved and FileCheck checks fail.
config.environment['PYTHONUNBUFFERED'] = '1'
