#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import os
import tempfile
import shutil
from pathlib import Path
from .utils import print_command, run_and_save, run_command, add_extension
from .library.blas import gemm

def build_main_obj(benchmark_prog, m, n, k, op, reps, mktmp_fn):
    benchmark_prog = benchmark_prog.replace("_M_", str(m))
    benchmark_prog = benchmark_prog.replace("_K_", str(k))
    benchmark_prog = benchmark_prog.replace("_N_", str(n))
    benchmark_prog = benchmark_prog.replace("__OP__", op)
    benchmark_prog = benchmark_prog.replace("_REPS_", str(reps))

    main_mlir = mktmp_fn("test.mlir")
    main_mlir_lowered = mktmp_fn("test.llvm.mlir")
    main_llvm = mktmp_fn("test.ll")
    main_obj = mktmp_fn("test.o")

    f = open(main_mlir, "w")
    f.write(benchmark_prog)
    f.close()

    # main program
    cmd = ["mlir-opt"]
    cmd.append(main_mlir)
    cmd.append("--linalg-bufferize")
    cmd.append("--std-bufferize")
    cmd.append("--tensor-constant-bufferize")
    cmd.append("--tensor-bufferize")
    cmd.append("--func-bufferize")
    cmd.append("-convert-linalg-to-affine-loops")
    cmd.append("-lower-affine")
    cmd.append("-convert-scf-to-cf")
    cmd.append("-convert-memref-to-llvm")
    cmd.append("-convert-std-to-llvm")
    cmd.append("-reconcile-unrealized-casts")
    cmd.append(f"> {main_mlir_lowered}")
    run_command(cmd)
    print_command(cmd)

    cmd = ["mlir-translate"]
    cmd.append("--mlir-to-llvmir")
    cmd.append(f"{main_mlir_lowered}")
    cmd.append(f"> {main_llvm}")
    run_command(cmd)

    cmd = ["llc"]
    cmd.append(f"{main_llvm}")
    cmd.append("-O3")
    cmd.append("-filetype=obj")
    cmd.append(f"-o {main_obj}")
    run_command(cmd)


def apply(transform_list, op_mlir_file, verbosity_level):
    cmd = ["$IREE_LLVM_SANDBOX_BUILD_DIR/bin/mlir-proto-opt "]
    for t in transform_list:
      if not t:
        continue
      if type(t) is tuple:
        (l, ext) = t
        if l >= verbosity_level:
          run_and_save(cmd, op_mlir_file, add_extension(op_mlir_file, ext))
      else:
        cmd.append(t)

    output = add_extension(op_mlir_file, "llvm")
    run_and_save(cmd, op_mlir_file, output)
    return output

def SaveIR(x, ext):
  return (x, ext)

def build_operator_obj(op_prog, m, n, k, op, option_list, mktmp_fn, verbosity_level=0):
    op_prog = op_prog.replace("_M_", str(m))
    op_prog = op_prog.replace("_K_", str(k))
    op_prog = op_prog.replace("_N_", str(n))

    op_mlir = mktmp_fn(f"{op}.mlir")
    f = open(f"{op_mlir}", "w")
    f.write(op_prog)
    f.close()

    # Transformation options
    tile_sizes = option_list["tile_sizes"]
    reorder_tile_sizes = option_list["reorder_tile_sizes"]
    register_tile_sizes = option_list["register_tile_sizes"]
    reorder_register_tile_sizes = option_list["reorder_register_tile_sizes"]
    hoist_packing = option_list['hoist_packing']
    split_vector_transfer = option_list['split_vector_transfers_to']
    extract_micro_kernel = option_list['extract_micro_kernel']
    modulo_scheduling = option_list['modulo_scheduling']

    Canonicalize = " --canonicalize --cse"
    CodegenDriver = "--linalg-tensor-codegen-driver=\"anchor-func=gemm anchor-op=linalg.generic"
    Tile = "--linalg-single-tiling-expert-driver=\"anchor-func=gemm anchor-op=linalg.generic"

    # Transformations
    OuterTiling = Tile + f" tile-sizes={tile_sizes} tile-interchange={reorder_tile_sizes}\"" + Canonicalize

    InnerTiling = Tile + f" tile-sizes={register_tile_sizes} tile-interchange={reorder_register_tile_sizes}" + \
                                  f" pad pack-paddings=1,1,0 hoist-paddings={hoist_packing} \"" + Canonicalize

    DecomposeToLowerDimensionalNamedOp = Tile + " decompose-to-lower-dim\"" + Canonicalize

    Vectorize = Tile + " vectorize vectorize-padding\"" + Canonicalize

    Bufferize = "--linalg-bufferization-driver" + Canonicalize

    LowerVector = "--linalg-vector-lowering=\"max-transfer-rank=1 " +\
                 f" split-transfers={split_vector_transfer}" +\
                  " lower-vector-transpose-to=eltwise" +\
                  " lower-vector-multi-reduction-to=innerparallel" +\
                  " lower-vector-contraction-to=outerproduct" +\
                  " unroll-vector-transfers=true"

    LowerVectorStage = lambda stage : LowerVector+f" lower-vector-stage={stage}\"" + Canonicalize

    ExtractKernel = "--alp-extract-kernel" + Canonicalize if extract_micro_kernel else ""
    ModuloScheduling = "--alp-modulo-scheduling=\"interleave unrolling=16\"" + Canonicalize  if modulo_scheduling else "" # TODO: Order is not preserved if I canonicalize
    Legalize = "--alp-legalize" + Canonicalize
    ExtractKernelTail = "--alp-extract-kernel-tail" + Canonicalize

    LowerToLLVM = "--convert-vector-to-scf " +\
                  "--convert-linalg-to-loops " +\
                  "--canonicalize " +\
                  "--lower-affine " +\
                  "--convert-scf-to-cf " +\
                  "--convert-linalg-to-llvm " +\
                  "--convert-vector-to-llvm " +\
                  "--convert-math-to-llvm " +\
                  "--convert-memref-to-llvm " +\
                  "--convert-std-to-llvm " +\
                  "--canonicalize " +\
                  "--cse " +\
                  "--reconcile-unrealized-casts "


    TransformList = [OuterTiling,
                     InnerTiling,
                     SaveIR(4, "tile"),
                     DecomposeToLowerDimensionalNamedOp,
                     Vectorize,
                     SaveIR(4, "vectorize"),
                     Bufferize,
                     ExtractKernel,
                     SaveIR(4, "bufferize"),
                     Legalize,
                     SaveIR(4, "legalize"),
                     SaveIR(4, "micro_kernel"),
                     LowerVectorStage(0),
                     SaveIR(4, "micro_kernel_2"),
                     LowerVectorStage(1),
                     LowerVectorStage(2),
                     LowerVectorStage(3),
                     LowerVectorStage(4),
                     LowerVectorStage(5),
                     LowerVectorStage(6),
                     ModuloScheduling,
                     ExtractKernelTail,
                     SaveIR(4, "micro_kernel_final"),
                     LowerToLLVM]

    op_llvm_mlir = apply(TransformList, op_mlir, verbosity_level)

    out = run_command(["$IREE_LLVM_SANDBOX_BUILD_DIR/bin/mlir-translate --mlir-to-llvmir " + op_llvm_mlir])
    op_llvm = mktmp_fn(f"{op}.ll")
    f = open(f"{op_llvm}", "w")
    f.write(out)
    f.close()

    op_obj = mktmp_fn(f"{op}.o")
    op_asm = mktmp_fn(f"{op}.s")

    cmd = ["llc"]
    cmd.append(op_llvm)
    cmd.append("-O1")
    cmd.append("-regalloc=greedy")
    cmd.append("-filetype=obj")
    cmd.append(f"-o {op_obj}")
    run_command(cmd)

    cmd = ["llc"]
    cmd.append(f"{op_llvm}")
    cmd.append("-O1")
    cmd.append("-regalloc=greedy")
    cmd.append("-filetype=asm")
    cmd.append(f"-o {op_asm}")
    run_command(cmd)

def link_main(op, mktmp_fn):
    out_bin = "exec_matmul"
    main_obj = mktmp_fn("test.o")
    op_obj = mktmp_fn(f"{op}.o")

    cmd = ["clang++"]
    cmd.append(f"{main_obj}")
    cmd.append(f"{op_obj}")
    cmd.append("$IREE_LLVM_SANDBOX_SOURCE_DIR/experimental/alp/lib/AlpRuntime/alp_runtime.cpp")
    cmd.append(f"-o {out_bin}")
    cmd.append("-lmlir_c_runner_utils")
    print_command(cmd)
    run_command(cmd)

def build_mlir(op, m, n, k, options):
    verbose = ("verbosity_level" in options) and options["verbosity_level"] > 0
    reps= 1
    if options["reps"]:
      reps = options["reps"]

    if verbose:
      Path("./tmp").mkdir(exist_ok=True)
      tmp_dir_name = "./tmp"
      verbosity_level=options["verbosity_level"]
    else:
      tmp_dir = tempfile.TemporaryDirectory()
      tmp_dir_name = tmp_dir.name
      verbosity_level=0

    (benchmark, op_mlir)= gemm(trA=True)
    mktmp = lambda x : os.path.join(tmp_dir_name, x)
    build_main_obj(benchmark, m, n, k, op, reps, mktmp)
    build_operator_obj(op_mlir, m, n, k, op, options, mktmp, verbosity_level)
    link_main(op, mktmp)
