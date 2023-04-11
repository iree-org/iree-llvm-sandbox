#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import tempfile
import shutil
import argparse
from pathlib import Path

# Alp imports
from ..backend.utils import print_command, run_and_save, run_command, add_extension
from ..transition.blas.gemm import save_mlir
from ..transition.blas.gemm import GEMM, matmul_NN, matmul_TN


def compile(mlir_test):
  mlir_test_base = os.path.splitext(mlir_test)[0]
  mlir_llvm_test = mlir_test_base + ".llvm.mlir"
  llvm_test = mlir_test_base + ".ll"
  obj_test = mlir_test_base + ".o"

  # main program
  cmd = ["$IREE_LLVM_SANDBOX_BUILD_DIR/bin/mlir-opt"]
  cmd.append(mlir_test)
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
  cmd.append(f"> {mlir_llvm_test}")
  run_command(cmd)

  cmd = ["$IREE_LLVM_SANDBOX_BUILD_DIR/bin/mlir-translate"]
  cmd.append("--mlir-to-llvmir")
  cmd.append(f"{mlir_llvm_test}")
  cmd.append(f"> {llvm_test}")
  run_command(cmd)

  cmd = ["$IREE_LLVM_SANDBOX_BUILD_DIR/bin/llc"]
  cmd.append(f"{llvm_test}")
  cmd.append("-O3")
  cmd.append("-filetype=obj")
  cmd.append(f"-o {obj_test}")
  run_command(cmd)
  return obj_test


def link(asm_program, obj_test):
  mlir_test_base = os.path.splitext(obj_test)[0]
  exe_test = mlir_test_base + ".out"

  runtime_src = (
      "$IREE_LLVM_SANDBOX_SOURCE_DIR/experimental/alp/lib/AlpRuntime/alp_runtime.cpp"
  )
  cmd = ["$IREE_LLVM_SANDBOX_BUILD_DIR/bin/clang++"]
  cmd.append(f"{obj_test}")
  cmd.append(f"{asm_program}")
  cmd.append(runtime_src)
  cmd.append(f"-o {exe_test}")
  run_command(cmd)
  return exe_test


def build(asm_program, mlir_test):
  # Compile and link
  obj_test = compile(mlir_test)
  return link(asm_program, obj_test)


def run(executable):
  run_command([os.path.abspath(executable)])
