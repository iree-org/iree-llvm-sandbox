#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import os
import argparse
from pathlib import Path

# ALP specifics imports
from .utils import run_and_save, run_command, add_extension
from .codegen import codegen
from .transforms import Pipeline, ExtractKernel, ConvertLoops

# Sandbox imports
import mlir.iree_sandbox as sandbox
import iree.compiler.ir as ir
import iree.compiler.dialects.linalg_transform as transform
from mlir.iree_sandbox import register_sandbox_passes_and_dialects
from examples.core.transforms import *
from examples.core.transform import (
    TransformListFactory,
    TransformationList,
    PrintIR,
    SaveIR,
)

# Standalone
from iree.compiler.ir import *
from iree.compiler.dialects import arith, builtin, linalg, tensor, scf, func, memref
from iree.compiler.dialects.linalg.opdsl.lang import *

Tiling = Tile.then(Tile).then(Vectorize).then(Bufferize)


def apply(transform, source, dest, verbosity_level=0):
  sourcef = open(source)
  destf = open(dest, "w")
  mlir_asm = sourcef.read()

  with Context() as ctx:
    register_sandbox_passes_and_dialects(ctx)
    module = Module.parse(asm=mlir_asm)
    out = transform("gemm", module)
    module_transformed = str(module)
  destf.write(module_transformed)


def translate_to_llvm(source, dest):
  out = run_command([
      "$IREE_LLVM_SANDBOX_BUILD_DIR/bin/mlir-translate --mlir-to-llvmir " +
      source
  ])
  f = open(dest, "w")
  f.write(out)
  f.close()


def generate_interchange_2d(transpose_flags):

  def gen_map(flag):
    id_map = [0, 1]
    if flag:
      return id_map[::-1]
    return id_map

  return map(gen_map, transpose_flags)


def generate_transform_pipeline(options):

  # Transformation options
  tile_sizes = options["tile_sizes"]
  reorder_tile_sizes = options["reorder_tile_sizes"]
  register_tile_sizes = options["register_tile_sizes"]
  reorder_register_tile_sizes = options["reorder_register_tile_sizes"]
  hoist_packing = options["hoist_packing"]
  split_vector_transfer = options["split_vector_transfers_to"]
  extract_micro_kernel = options["extract_micro_kernel"]
  modulo_scheduling = options["modulo_scheduling"]
  ms_unroll = options["ms_unroll"] if options["ms_unroll"] else 2
  ms_distance = options["ms_distance"] if options["ms_distance"] else 1
  transpose_packing = options["transpose_packing"]

  tile = Tiling(
      "gemm",
      "linalg.generic",
      tile_sizes1=tile_sizes,
      tile_interchange1=reorder_tile_sizes,
      tile_sizes2=register_tile_sizes,
      tile_interchange2=reorder_register_tile_sizes,
      pad2=True,
      pack_paddings2=[1, 1, 0],
      hoist_paddings2=hoist_packing,
      transpose_paddings2=generate_interchange_2d(transpose_packing),
      vectorize_padding=True,
  )

  # Compose the MLIR pipeline
  transf = tile
  if extract_micro_kernel:
    transf = transf.then(ExtractKernel("gemm", "linalg.generic"))
  transf = transf.then(LowerVectors(split_transfers=split_vector_transfer))
  if modulo_scheduling:
    transf = transf.then(
        Pipeline("gemm",
                 "linalg.generic",
                 unroll=ms_unroll,
                 distance=ms_distance))

  transf = (transf.then(ConvertLoops("gemm", "linalg.generic")).then(
      ConvertLoops("kernel", "linalg.generic")).then(LowerToLLVM()))

  return transf


def compile(mlir_program, option_list):
  """The compiler program receives an mlir_program (.mlir) and generates
    assembly (.s)
    """
  program_base = os.path.splitext(mlir_program)[0]
  transformed_program = f"{program_base}.llvm.mlir"
  llvm_program = f"{program_base}.ll"
  asm_program = f"{program_base}.s"

  ## Transform the MLIR program
  # Generate a pipeline to transform the program
  pipeline = generate_transform_pipeline(option_list)

  # Add SaveIR transforms after each transformation
  if option_list["verbosity_level"] > 1:
    pipeline = pipeline.save_ir(file_name=mlir_program, after_all=True)

  # Apply the pipeline
  apply(pipeline, mlir_program, transformed_program)

  ## Translate MLIR LLVM to LLVMIR
  translate_to_llvm(transformed_program, llvm_program)

  ## MLIR part is over. Let's pass the ball to the code generator
  scheduler = option_list["scheduler"]
  codegen(llvm_program, asm_program, scheduler)

  return asm_program


if __name__ == "__main__":

  parser = argparse.ArgumentParser("mlirc")

  # Input MLIR to compile
  parser.add_argument("--input-file")

  # Outer tiling
  parser.add_argument("--tile-sizes", nargs="+", type=int)
  parser.add_argument("--reorder-tile-sizes", nargs="+", type=int)

  # Inner tiling
  parser.add_argument("--register-tile-sizes", nargs="+", type=int)
  parser.add_argument("--reorder-register-tile-sizes", nargs="+", type=int)
  parser.add_argument("--hoist-packing", nargs="+", type=int)
  parser.add_argument("--transpose-packing", nargs="+", type=int)

  # Vector lowering
  parser.add_argument("--unroll-vector-transfers", action="store_true")
  parser.add_argument("--split-vector-transfers-to")

  # micro-kernel transforms
  parser.add_argument("--extract-micro-kernel", action="store_true")
  parser.add_argument("--modulo-scheduling", action="store_true")
  parser.add_argument("--ms-interleave", action="store_true")
  parser.add_argument("--ms-unroll", type=int)
  parser.add_argument("--ms-distance", type=int)

  # scheduling algorithm
  parser.add_argument("--scheduler", default="ilpmax")

  # verbosity
  parser.add_argument("--verbosity-level", type=int, default=4)
  args = parser.parse_args()

  options = {
      "tile_sizes": args.tile_sizes,
      "register_tile_sizes": args.register_tile_sizes,
      "split_vector_transfers_to": args.split_vector_transfers_to,
      "unroll_vector_transfers": args.unroll_vector_transfers,
      "reorder_tile_sizes": args.reorder_tile_sizes,
      "reorder_register_tile_sizes": args.reorder_register_tile_sizes,
      "hoist_packing": args.hoist_packing,
      "transpose_packing": args.transpose_packing,
      "extract_micro_kernel": args.extract_micro_kernel,
      "modulo_scheduling": args.modulo_scheduling,
      "ms_interleave": args.ms_interleave,
      "ms_unroll": args.ms_unroll,
      "ms_distance": args.ms_distance,
      "scheduler": args.scheduler,
      "verbosity_level": args.verbosity_level,
  }

  compile(args.input_file, options)
