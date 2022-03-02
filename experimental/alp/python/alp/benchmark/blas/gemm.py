#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys, time, os

from typing import Any, List, Mapping, Optional, Sequence
import numpy as np
import argparse

from mlir.ir import *
from mlir.dialects import arith, builtin, linalg, tensor, scf, func, memref
from mlir.dialects.linalg.opdsl.lang import *

from ...transition.blas.gemm import GEMM
from ..infra import *

from examples.core.problem_definition import *


def save_mlir(mlir_txt, dest):
  f = open(dest, "w")
  f.write(mlir_txt)
  f.close()


def emit_benchmarking_function(trA, sizes, niter,
                               func: builtin.FuncOp) -> builtin.FuncOp:
  """Produces the benchmarking function.

    This function calls the given function `func` as many times as requested by
    its last argument.
    """
  f64 = F64Type.get()

  print_flops = builtin.FuncOp("print_flops", ([f64], []), visibility="private")

  print_time = builtin.FuncOp("print_time", ([f64], []), visibility="private")

  print_pid = builtin.FuncOp("print_pid", ([], []), visibility="private")

  rtclock = builtin.FuncOp("rtclock", ([], [f64]), visibility="private")

  wrapper = builtin.FuncOp(
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
    one = arith.ConstantOp.create_index(1)
    elem = arith.ConstantOp(F32Type.get(), 1.0)
    nops = arith.ConstantOp(F64Type.get(), 1.0 * (2 * M * N * K))

    if trA:
      A0 = linalg.InitTensorOp([K, M], F32Type.get())
    else:
      A0 = linalg.InitTensorOp([M, K], F32Type.get())

    B0 = linalg.InitTensorOp([K, N], F32Type.get())
    C = linalg.InitTensorOp([M, N], F32Type.get())

    A = linalg.FillOp(output=A0.results[0], value=elem)
    B = linalg.FillOp(output=B0.results[0], value=elem)
    func.CallOp(print_pid, [])

    call = func.CallOp(func, [A.results[0], B.results[0], C.results[0]])

    n_iterations = arith.ConstantOp.create_index(niter)
    start = func.CallOp(rtclock, [])
    loop = scf.ForOp(zero, n_iterations, one, [])
    with InsertionPoint(loop.body):
      call = func.CallOp(func, [A.results[0], B.results[0], C.results[0]])
      scf.YieldOp([])
    end = func.CallOp(rtclock, [])
    treps = arith.SubFOp(end, start)
    n_iterations_f = arith.ConstantOp(F64Type.get(), float(niter))
    t = arith.DivFOp(treps, n_iterations_f)
    flops = arith.DivFOp(nops, t)
    func.CallOp(print_time, [t.results[0]])
    func.CallOp(print_flops, [flops.results[0]])

    ret = arith.ConstantOp(IntegerType.get_signless(32), 0)
    func.ReturnOp(ret)

  return wrapper


def generate_benchmark_mlir(func_name, trA, size, reps, dest):
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
      gemm = builtin.FuncOp(func_name, (types, [types[-1]]),
                            visibility="private")
      benchmark = emit_benchmarking_function(trA, size, reps, gemm)
  save_mlir(str(mlir_module), dest)


def main(argv):
  parser = argparse.ArgumentParser("gemm")

  # Files and paths
  parser.add_argument("--asm-program", default="")
  parser.add_argument("--function-name", default="gemm")
  parser.add_argument("--output", default="")

  # GEMM specific parameters for the problem to benchmark
  parser.add_argument("--M", type=int)
  parser.add_argument("--N", type=int)
  parser.add_argument("--K", type=int)
  parser.add_argument("--trA", action="store_true")

  # Benchmark specs
  parser.add_argument("--reps", type=int, default=1)
  args = parser.parse_args(argv)

  # Generate the benchmark, build and run
  size = [args.M, args.N, args.K]
  dest = args.output if args.output else args.function_name + ".bench.mlir"
  generate_benchmark_mlir(args.function_name, args.trA, size, args.reps, dest)

  if args.asm_program:
    exe = build(args.asm_program, dest)
    print(run(exe))


if __name__ == "__main__":
  main(os.sys.argv[1:])
