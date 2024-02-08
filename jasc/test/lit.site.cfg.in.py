# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

@LIT_SITE_CFG_IN_HEADER@

import os.path

config.llvm_tools_dir = lit_config.substitute("@LLVM_TOOLS_DIR@")
config.llvm_shlib_ext = "@SHLIBEXT@"
config.llvm_shlib_dir = lit_config.substitute(path(r"@SHLIBDIR@"))
config.mlir_tools_dir = "@MLIR_TOOLS_DIR@"
config.jasc_src_dir = "@JASC_SOURCE_DIR@"
config.jasc_tools_dir = "@JASC_TOOLS_DIR@"

import lit.llvm
lit.llvm.initialize(lit_config, config)

# Let the main config do the real work.
lit_config.load_config(config, os.path.join(config.jasc_src_dir, "test/lit.cfg.py"))
