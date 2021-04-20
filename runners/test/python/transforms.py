# Bootstrap our local extensions first.
# TODO: Requires that both ${LLVM_INSTALL}/python and ./build are on
# PYTHONPATH
import runners

import time
from mlir.ir import *
from mlir.passmanager import *


def fuse(module: Module,
         func_name: str,
         op_name: str,
         tile_sizes: list,
         pad=False):
  pad_str = f'fuse-padding' if pad else ''
  tile_str = f'tile-sizes={",".join([str(ts) for ts in tile_sizes])}'
  pipeline = (f'func(linalg-tensor-codegen-strategy{{anchor-func={func_name} '
              f'     anchor-op={op_name} '
              f'     fuse '
              f'     {pad_str}'
              f'     {tile_str}}}),'
              f'canonicalize,'
              f'cse')
  PassManager.parse(pipeline).run(module)


def tile_and_pad(module: Module,
                 func_name: str,
                 op_name: str,
                 tile_sizes: list,
                 pad=False,
                 hoist_padding=None):
  pad_str, hoist_padding_str = '', ''
  tile_str = f'tile-sizes={",".join([str(ts) for ts in tile_sizes])}'
  if pad:
    pad_str = 'pad'
  if hoist_padding:
    hoist_padding_str = f'hoist-padding={hoist_padding}'
  pipeline = (f'func(linalg-tensor-codegen-strategy{{anchor-func={func_name} '
              f'     anchor-op={op_name} '
              f'     {tile_str} '
              f'     {pad_str} '
              f'     {hoist_padding_str}}}),'
              f'canonicalize,'
              f'cse')
  PassManager.parse(pipeline).run(module)


def vectorize(module: Module, func_name: str, op_name: str):
  pipeline = (f'func(linalg-tensor-codegen-strategy{{anchor-func={func_name} '
              f'     anchor-op={op_name} '
              f'     vectorize '
              f'     vectorize-padding}}),'
              f'canonicalize,'
              f'cse')
  PassManager.parse(pipeline).run(module)


def lower_to_llvm(module: Module):
  pipeline = (f'linalg-comprehensive-bufferize-inplace,'
              f'func(convert-linalg-to-loops,'
              f'     convert-vector-to-scf{{full-unroll=true}}),'
              f'canonicalize,'
              f'cse,'
              f'lower-affine,'
              f'convert-scf-to-std,'
              f'convert-vector-to-llvm,'
              f'convert-std-to-llvm')
  PassManager.parse(pipeline).run(module)


def bufferize(module: Module):
  pipeline = (f'linalg-comprehensive-bufferize-inplace,'
              f'canonicalize,'
              f'cse')
  PassManager.parse(pipeline).run(module)


def pre_transform(module, boilerplate_code):
  import mlir.conversions
  import mlir.dialects.linalg.passes
  import mlir.transforms

  # TODO: Allow cloning functions from one module to another.
  # Atm we have to resort to string concatenation.
  module = Module.parse(
      str(module.operation.regions[0].blocks[0].operations[0].operation) +
      boilerplate_code)

  return module


def expert_compilerr_1(module, boilerplate_code):
  module = pre_transform(module, boilerplate_code)
  tile_and_pad(module, 'matmul_on_tensors', 'linalg.matmul', [256, 256, 256])
  tile_and_pad(module, 'matmul_on_tensors', 'linalg.matmul', [64, 64, 64])
  tile_and_pad(
      module,
      'matmul_on_tensors',
      'linalg.matmul', [8, 16, 32],
      pad=True,
      hoist_padding=2)
  vectorize(module, 'matmul_on_tensors', 'linalg.matmul')
  bufferize(module)
  lower_to_llvm(module)
  return module


def expert_compilerr_2(module, boilerplate_code):
  module = pre_transform(module, boilerplate_code)
  fuse(module, 'matmul_on_tensors', 'linalg.matmul', [256, 256])
  fuse(module, 'matmul_on_tensors', 'linalg.matmul', [8, 16])
  tile_and_pad(module, 'matmul_on_tensors', 'linalg.matmul', [0, 0, 32])
  vectorize(module, 'matmul_on_tensors', 'linalg.matmul')
  vectorize(module, 'matmul_on_tensors', 'linalg.fill')
  bufferize(module)
  lower_to_llvm(module)
  return module


def expert_compilerr_3(module, boilerplate_code):
  module = pre_transform(module, boilerplate_code)
  fuse(module, 'matmul_on_tensors', 'linalg.matmul', [256, 256])
  tile_and_pad(
      module,
      'matmul_on_tensors',
      'linalg.matmul', [8, 16, 32],
      pad=True,
      hoist_padding=3)
  vectorize(module, 'matmul_on_tensors', 'linalg.matmul')
  tile_and_pad(module, 'matmul_on_tensors', 'linalg.fill', [8, 32])
  vectorize(module, 'matmul_on_tensors', 'linalg.fill')
  bufferize(module)
  lower_to_llvm(module)
  return module
