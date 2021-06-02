# Bootstrap our local extensions first.
# TODO: Requires that both ${LLVM_INSTALL}/python and ./build are on
# PYTHONPATH

import runners

from mlir.ir import *
from mlir.passmanager import *

import mlir.all_passes_registration


class Transform:
  """Base class for all parametrized transformations."""

  def __call__(self, module: Module, func_name: str):
    PassManager.parse(self.pipeline).run(module)


class Fuse(Transform):

  def __init__(self, func_name: str, op_name: str, tile_sizes: list, pad=False):
    pad_str = f'fuse-padding' if pad else ''
    tile_str = f'tile-sizes={",".join([str(ts) for ts in tile_sizes])}'
    pipeline = (f'func(linalg-tensor-codegen-strategy{{anchor-func={func_name} '
                f'     anchor-op={op_name} '
                f'     fuse '
                f'     {pad_str}'
                f'     {tile_str}}}),'
                f'canonicalize,'
                f'cse')
    self.pipeline = pipeline


class TileAndPad(Transform):

  def __init__(self,
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
    self.pipeline = pipeline


class Vectorize(Transform):

  def __init__(self, func_name: str, op_name: str):
    pipeline = (f'func(linalg-tensor-codegen-strategy{{anchor-func={func_name} '
                f'     anchor-op={op_name} '
                f'     vectorize '
                f'     vectorize-padding}}),'
                f'canonicalize,'
                f'cse')
    self.pipeline = pipeline


class Bufferize(Transform):

  def __init__(self):
    pipeline = (f'linalg-comprehensive-bufferize-inplace,'
                f'canonicalize,'
                f'cse')
    self.pipeline = pipeline


class LowerToLLVM(Transform):

  def __init__(self):
    pipeline = (f'linalg-comprehensive-bufferize-inplace,'
                f'func(convert-linalg-to-loops,'
                f'     convert-vector-to-scf{{full-unroll=true}}),'
                f'canonicalize,'
                f'cse,'
                f'lower-affine,'
                f'convert-scf-to-std,'
                f'convert-vector-to-llvm,'
                f'convert-std-to-llvm')
    self.pipeline = pipeline


class Sparsify(Transform):

  def __init__(self, options: str):
    pipeline = (
        f'sparsification{{{options}}},'
        f'sparse-tensor-conversion,'
        f'func(convert-linalg-to-loops,convert-vector-to-scf),'
        f'convert-scf-to-std,'
        f'func-bufferize,'
        f'tensor-constant-bufferize,'
        f'func(tensor-bufferize,std-bufferize,finalizing-bufferize),'
        f'convert-vector-to-llvm{{reassociate-fp-reductions=1 enable-index-optimizations=1}},'
        f'convert-std-to-llvm')
    self.pipeline = pipeline
