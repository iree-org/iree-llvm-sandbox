# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# -*- Python -*-

import os

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' test runner.

config.name = "Jasc"

config.test_format = lit.formats.ShTest(execute_external=False)

config.suffixes = [
    ".mlir",
]

config.excludes = [
    "lit.cfg.py",
]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_source_root)

llvm_config.use_default_substitutions()

tool_dirs = [
    config.jasc_tools_dir,
    config.mlir_tools_dir,
    config.llvm_tools_dir,
]
tools = [
    "jasc-opt",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
