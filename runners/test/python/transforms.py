# Bootstrap our local extensions first.
# TODO: Requires that both ${LLVM_INSTALL}/python and ./build are on
# PYTHONPATH
import runners

import time
from mlir.ir import *
from mlir.passmanager import *
import mlir.conversions
import mlir.dialects.linalg.passes
import mlir.transforms


class Expert:

  def __init__(self, transforms):
    self.transforms = transforms

  def _pre_transform(self, module, boilerplate_code):

    # TODO: Allow cloning functions from one module to another.
    # Atm we have to resort to string concatenation.
    module = Module.parse(
        str(module.operation.regions[0].blocks[0].operations[0].operation) +
        boilerplate_code)

    return module

  def __call__(self, module, boilerplate_code):
    module = self._pre_transform(module, boilerplate_code)
    for transform in self.transforms:
      transform(module, 'matmul_on_tensors')
    return module


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


expert_compilerr_1 = Expert([
    TileAndPad('matmul_on_tensors', 'linalg.matmul', [256, 256, 256]),
    TileAndPad('matmul_on_tensors', 'linalg.matmul', [64, 64, 64]),
    TileAndPad(
        'matmul_on_tensors',
        'linalg.matmul', [8, 16, 32],
        pad=True,
        hoist_padding=2),
    Vectorize('matmul_on_tensors', 'linalg.matmul'),
    Bufferize(),
    LowerToLLVM(),
])

expert_compilerr_2 = Expert([
    Fuse('matmul_on_tensors', 'linalg.matmul', [256, 256]),
    Fuse('matmul_on_tensors', 'linalg.matmul', [8, 16]),
    TileAndPad('matmul_on_tensors', 'linalg.matmul', [0, 0, 32]),
    Vectorize('matmul_on_tensors', 'linalg.matmul'),
    Vectorize('matmul_on_tensors', 'linalg.fill'),
    Bufferize(),
    LowerToLLVM(),
])

expert_compilerr_3 = Expert([
    Fuse('matmul_on_tensors', 'linalg.matmul', [256, 256]),
    TileAndPad(
        'matmul_on_tensors',
        'linalg.matmul', [8, 16, 32],
        pad=True,
        hoist_padding=3),
    Vectorize('matmul_on_tensors', 'linalg.matmul'),
    TileAndPad('matmul_on_tensors', 'linalg.fill', [8, 32]),
    Vectorize('matmul_on_tensors', 'linalg.fill'),
    Bufferize(),
    LowerToLLVM(),
])
