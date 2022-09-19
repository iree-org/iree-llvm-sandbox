# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os
import subprocess

import lit.formats

# name: The name of this test suite.
config.name = 'DatabaseIterators Unit'

# suffixes: A list of file extensions to treat as test files.
config.suffixes = []

# test_source_root: The root path where tests are located.
# test_exec_root: The root path where tests should be run.
config.test_exec_root = '@CMAKE_CURRENT_BINARY_DIR@'
config.test_source_root = config.test_exec_root

# Pass through LLVM_SYMBOLIZER_PATH from environment
if "LLVM_SYMBOLIZER_PATH" in os.environ:
  config.environment["LLVM_SYMBOLIZER_PATH"] = \
      os.environ["LLVM_SYMBOLIZER_PATH"]

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.GoogleTest('.', 'Tests')
