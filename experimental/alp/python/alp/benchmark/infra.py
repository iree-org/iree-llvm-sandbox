#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import tempfile
import shutil
import argparse
from pathlib import Path
from ..backend.utils import print_command, run_and_save, run_command, add_extension


def compile(mlir_benchmark):
  mlir_benchmark_base = os.path.splitext(mlir_benchmark)[0]
  mlir_llvm_benchmark = mlir_benchmark_base + ".llvm.mlir"
  llvm_benchmark = mlir_benchmark_base + ".ll"
  obj_benchmark = mlir_benchmark_base + ".o"

  # main program
  cmd = ["$IREE_LLVM_SANDBOX_BUILD_DIR/bin/mlir-opt"]
  cmd.append(mlir_benchmark)
  cmd.append("--linalg-bufferize")
  cmd.append("--std-bufferize")
  cmd.append("--tensor-bufferize")
  cmd.append("--func-bufferize")
  cmd.append("-convert-linalg-to-affine-loops")
  cmd.append("-lower-affine")
  cmd.append("-convert-scf-to-std")
  cmd.append("-convert-memref-to-llvm")
  cmd.append("-convert-std-to-llvm")
  cmd.append("-reconcile-unrealized-casts")
  cmd.append(f"> {mlir_llvm_benchmark}")
  run_command(cmd)

  cmd = ["$IREE_LLVM_SANDBOX_BUILD_DIR/bin/mlir-translate"]
  cmd.append("--mlir-to-llvmir")
  cmd.append(f"{mlir_llvm_benchmark}")
  cmd.append(f"> {llvm_benchmark}")
  run_command(cmd)

  cmd = ["$IREE_LLVM_SANDBOX_BUILD_DIR/bin/llc"]
  cmd.append(f"{llvm_benchmark}")
  cmd.append("-O3")
  cmd.append("-filetype=obj")
  cmd.append(f"-o {obj_benchmark}")
  run_command(cmd)
  return obj_benchmark


def link(asm_program, obj_benchmark):
  mlir_benchmark_base = os.path.splitext(obj_benchmark)[0]
  exe_benchmark = mlir_benchmark_base + ".out"

  runtime_src = (
      "$IREE_LLVM_SANDBOX_SOURCE_DIR/experimental/alp/lib/AlpRuntime/alp_runtime.cpp"
  )
  cmd = ["$IREE_LLVM_SANDBOX_BUILD_DIR/bin/clang++"]
  cmd.append(f"{obj_benchmark}")
  cmd.append(f"{asm_program}")
  cmd.append(runtime_src)
  cmd.append(f"-o {exe_benchmark}")
  run_command(cmd)
  return exe_benchmark


def build(asm_program, mlir_benchmark):
  # Compile and link
  obj_benchmark = compile(mlir_benchmark)
  return link(asm_program, obj_benchmark)


def run(executable):
  return run_command([os.path.abspath(executable)])
