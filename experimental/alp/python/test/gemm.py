#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from math import exp
import os
from pathlib import Path

import alp.backend.mlirc
import alp.benchmark.infra
import alp.benchmark.blas.gemm
import alp.test.blas.gemm
import alp.test.infra
import alp.backend.utils
import alp.transition.blas.gemm


def get_default_compiler_options():
  options = {
      "tile_sizes": [256, 256, 256],
      "register_tile_sizes": [4, 16, 1],
      "split_vector_transfers_to": "none",
      "unroll_vector_transfers": True,
      "reorder_tile_sizes": [0, 2, 1],
      "reorder_register_tile_sizes": [0, 1, 2],
      "hoist_packing": [4, 3, 0],
      "transpose_packing": [0, 0, 0],
      "extract_micro_kernel": False,
      "modulo_scheduling": False,
      "ms_unroll": 2,
      "ms_distance": 1,
      "scheduler": "ilpmax",
      "verbosity_level": 4,
  }
  return options


fun_name = "gemm"
mlir_name = f"{fun_name}.mlir"
bench_name = f"{fun_name}.bench.mlir"
test_name = f"{fun_name}.test.mlir"
dest_path = os.path.join(os.path.curdir, fun_name)
Path(dest_path).mkdir(exist_ok=True)


def gen_sgemm_trA():
  trA = True
  mlir_file_name = os.path.join(dest_path, mlir_name)
  # Generate a fully dynamic gemm
  alp.transition.blas.gemm.generate_mlir(fun_name, trA, [-1, -1, -1],
                                         mlir_file_name)
  asm_file = alp.backend.mlirc.compile(mlir_file_name,
                                       get_default_compiler_options())
  return asm_file


def perf(asm_file, sizes, trA, compare=False, expected_flops=[]):

  # Build GEMMTN for f32 with default compiler options

  # Benchmark for gemm 2048^3
  mlir_flops = []
  for size in sizes:
    bench_file_name = os.path.join(dest_path, bench_name)
    alp.benchmark.blas.gemm.generate_benchmark_mlir(fun_name, trA, size, 1,
                                                    bench_file_name)
    bench = alp.benchmark.infra.build(asm_file, bench_file_name)
    s, f = alp.backend.utils.parse(alp.benchmark.infra.run(bench))
    mlir_flops.append(f)

  if compare:
    for i in range(0, len(mlir_flops)):
      print(sizes[i], mlir_flops[i])

  slow_down = False
  for i in range(0, len(mlir_flops)):
    if expected_flops and mlir_flops[i] < expected_flops[i]:
      print(mlir_flops[i], expected_flops[i])
      slow_down = True
      break
  return slow_down


def verify(asm_file, sizes, trA):

  def verify(out):
    for l in out:
      if not l:
        continue
      f = float(l.strip())
      if f > 0.1:
        return False
    return True

  # Verify for very weird gemm dimensions (all prime numbers)
  for size in sizes:
    test_file_name = os.path.join(dest_path, test_name)
    alp.test.blas.gemm.generate_test_mlir(fun_name, trA, size, test_file_name)
    test = alp.test.infra.build(asm_file, test_file_name)
    out = alp.backend.utils.run_command([os.path.abspath(test)])
    out = out.split("\n")
    return True if verify(out) else False


def main():
  asm = gen_sgemm_trA()

  # Test performance against different sizes
  perf_sizes = [
      [2048, 2048, 2048],
      [1024, 1024, 1024],
      [512, 512, 512],
      [128, 128, 128],
      [64, 64, 64],
  ]
  slow_down = perf(asm,
                   perf_sizes,
                   compare=False,
                   trA=True,
                   expected_flops=[-1, -1, -1, -1, -1])
  if slow_down:
    print("PERF regression!!!")
  else:
    print("PERF OK!")

  ## Verify for weird sizes
  verify_sizes = [[513, 431, 23], [128, 10, 11]]
  correct = verify(asm, verify_sizes, trA=True)

  if correct:
    print("GEMM is correct!")
  else:
    print("Something is wrong!!!")


if __name__ == "__main__":
  main()
