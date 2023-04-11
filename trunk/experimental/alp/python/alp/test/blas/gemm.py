#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import argparse

# MLIR imports
from iree.compiler.ir import *
from iree.compiler.dialects import arith, func, linalg, tensor, scf, func, memref
from iree.compiler.dialects.linalg.opdsl.lang import *

# Sandbox imports
from examples.core.problem_definition import *

# Alp imports
from ...backend.utils import print_command, run_and_save, run_command, add_extension
from ...transition.blas.gemm import save_mlir
from ...transition.blas.gemm import GEMM, matmul_NN, matmul_TN
from ..infra import *


def emit_test_function(trA, sizes, func: func.FuncOp) -> func.FuncOp:
  """Produces the test function."""
  f64 = F64Type.get()
  f32 = F32Type.get()

  printF32 = func.FuncOp("printF32", ([f32], []), visibility="private")

  printNewline = func.FuncOp("printNewline", ([], []), visibility="private")
  wrapper = func.FuncOp(
      # Same signature and an extra buffer of indices to save timings.
      "main",
      ([], [IntegerType.get_signless(32)]),
      visibility="public",
  )

  M = sizes[0]
  N = sizes[1]
  K = sizes[2]

  with InsertionPoint(wrapper.add_entry_block()):
    zero = arith.ConstantOp.create_index(0)
    one_index = arith.ConstantOp.create_index(1)
    one_f64 = arith.ConstantOp(F64Type.get(), 1.0)

    cM = arith.ConstantOp.create_index(M)
    cN = arith.ConstantOp.create_index(N)
    cK = arith.ConstantOp.create_index(K)

    minus_one_f64 = arith.ConstantOp(F64Type.get(), -1.0)
    c12345_i32 = arith.ConstantOp(IntegerType.get_signless(32), 12345)
    c6789_i32 = arith.ConstantOp(IntegerType.get_signless(32), 6789)

    # Initialize tensors
    if trA:
      A0 = linalg.InitTensorOp([K, M], F32Type.get())
    else:
      A0 = linalg.InitTensorOp([M, K], F32Type.get())

    B0 = linalg.InitTensorOp([K, N], F32Type.get())
    C = linalg.InitTensorOp([M, N], F32Type.get())
    D = linalg.InitTensorOp([M, N], F32Type.get())

    elem = arith.ConstantOp(F32Type.get(), 1.0)
    # Fill the inputs
    A = linalg.fill_rng_2d(minus_one_f64,
                           one_f64,
                           c12345_i32,
                           outs=[A0.results[0]])
    B = linalg.fill_rng_2d(minus_one_f64,
                           one_f64,
                           c6789_i32,
                           outs=[B0.results[0]])
    # A = linalg.fill(elem, outs=[A0.results[0]])
    # B = linalg.fill(elem, outs=[B0.results[0]])

    # Evaluate actual function and the reference
    func.CallOp(func, [A, B, C.results[0]])
    if trA:
      D1 = matmul_TN(A, B, outs=[D])
    else:
      D1 = matmul_NN(A, B, outs=[D])

    # Verify correctness loop
    loopM = scf.ForOp(zero, cM, one_index, [])
    with InsertionPoint(loopM.body):
      loopN = scf.ForOp(zero, cN, one_index, [])
      with InsertionPoint(loopN.body):
        x = loopM.induction_variable
        y = loopN.induction_variable
        res1 = tensor.ExtractOp(F32Type.get(), C, [x, y])
        res2 = tensor.ExtractOp(F32Type.get(), D1, [x, y])
        diff = arith.SubFOp(res1, res2)
        # TODO Add support for scf.If op to verify directly from here
        func.CallOp(printF32, [diff.results[0]])
        func.CallOp(printNewline, [])
        scf.YieldOp([])
      scf.YieldOp([])

    ret = arith.ConstantOp(IntegerType.get_signless(32), 0)
    func.ReturnOp(ret)

  return wrapper


def generate_test_mlir(func_name, trA, size, dest):
  with Context() as ctx, Location.unknown() as loc:
    f32 = F32Type.get()
    problem_definition = GEMM(trA)
    mlir_module = Module.create()
    problem_sizes = {"M": size[0], "N": size[1], "K": size[2]}
    types = problem_definition.types_mlir_builder(
        problem_sizes,
        [f32, f32, f32],
    )

    with InsertionPoint(mlir_module.body):
      gemm = func.FuncOp(func_name, (types, [types[-1]]),
                            visibility="private")
      test = emit_test_function(trA, size, gemm)
  save_mlir(str(mlir_module), dest)


def main(argv):
  parser = argparse.ArgumentParser("benchmark")

  # Information about the program
  parser.add_argument("--asm-program")
  parser.add_argument("--function-name", default="gemm")
  parser.add_argument("--output", default="")

  # Details of the problem to test
  parser.add_argument("--trA", action="store_true")
  parser.add_argument("--M", type=int)
  parser.add_argument("--N", type=int)
  parser.add_argument("--K", type=int)

  # Test specs
  parser.add_argument("--threshold", type=float, default=0.1)
  args = parser.parse_args(argv)

  # Generate the test, build and run
  size = [args.M, args.N, args.K]
  dest = args.output if args.output else args.function_name + ".test.mlir"
  generate_test_mlir(args.function_name, args.trA, size, dest)
  exe = build(args.asm_program, dest)

  out = run_command([os.path.abspath(exe), "> out.log"])
  out = out.split("\n")
  for l in out:
    if not l:
      continue
    f = float(l.strip())
    if f > args.threshold:
      print("ERROR")
      exit()
  print("PASSED")


if __name__ == "__main__":
  main(os.sys.argv[1:])
