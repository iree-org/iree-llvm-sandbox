#
# import time

from typing import List
from search import *
from transforms import *

# Only for debugging: Print IR after each transform.
print_ir_after_each = False


class Assignments:

  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)


class Expert:

  def __init__(self, **asignments):
    self.assignments = Assignments(**asignments)

  def _pre_transform(self, module, boilerplate_code):
    benchmark_fun = str(
        module.operation.regions[0].blocks[0].operations[0].operation)
    module = Module.parse(benchmark_fun + boilerplate_code)
    return module

  def __call__(self, module, boilerplate_code):
    module = self._pre_transform(module, boilerplate_code)
    for transform in self.transforms():
      if print_ir_after_each:
        print('[[[ IR after transform: ' + str(transform) + ']]]')
      transform(module, 'matmul_on_tensors')
      if print_ir_after_each:
        print(module)
    return module

  def transforms(self) -> List[Transform]:
    'Abstract method that returns a list of transforms for given expert.'


class ExpertCompiler1(Expert):
  variables = {
      'sizes1': TilingSizesVariable,
      'sizes2': TilingSizesVariable,
      'sizes3': TilingSizesVariable,
      'pad': BoolVariable,
      'hoist_padding': HoistPaddingVariable,
  }

  def transforms(self) -> List[Transform]:
    v = self.assignments
    return [
        Tile('matmul_on_tensors', 'linalg.matmul', tile_sizes=v.sizes1),
        Tile('matmul_on_tensors', 'linalg.matmul', tile_sizes=v.sizes2),
        Tile(
            'matmul_on_tensors',
            'linalg.matmul',
            tile_sizes=v.sizes3,
            pad=v.pad,
            hoist_padding=v.hoist_padding),
        Vectorize('matmul_on_tensors', 'linalg.matmul'),
        Bufferize(),
        LowerToLLVM(),
    ]


class ExpertCompiler2(Expert):
  variables = {
      'sizes1': TilingSizesVariable,
      'sizes2': TilingSizesVariable,
      'sizes3': TilingSizesVariable,
  }

  def transforms(self) -> List[Transform]:
    v = self.assignments
    return [
        Fuse('matmul_on_tensors', 'linalg.matmul', tile_sizes=v.sizes1),
        Fuse('matmul_on_tensors', 'linalg.matmul', tile_sizes=v.sizes2),
        Tile('matmul_on_tensors', 'linalg.matmul', tile_sizes=v.sizes3),
        Vectorize('matmul_on_tensors', 'linalg.matmul'),
        Vectorize('matmul_on_tensors', 'linalg.fill'),
        Bufferize(),
        LowerToLLVM(),
    ]


class ExpertCompiler3(Expert):
  variables = {
      'sizes1': TilingSizesVariable,
      'sizes2': TilingSizesVariable,
      'sizes3': TilingSizesVariable,
      'pad': BoolVariable,
      'hoist_padding': HoistPaddingVariable,
  }

  def transforms(self) -> List[Transform]:
    v = self.assignments
    return [
        Fuse('matmul_on_tensors', 'linalg.matmul', tile_sizes=v.sizes1),
        Tile(
            'matmul_on_tensors',
            'linalg.matmul',
            tile_sizes=v.sizes2,
            pad=v.pad,
            hoist_padding=v.hoist_padding),
        Vectorize('matmul_on_tensors', 'linalg.matmul'),
        Tile('matmul_on_tensors', 'linalg.fill', tile_sizes=v.sizes3),
        Vectorize('matmul_on_tensors', 'linalg.fill'),
        Bufferize(),
        LowerToLLVM(),
    ]


class ExpertCompiler4(Expert):
  variables = {
      'sizes1': TilingSizesVariable,
      'sizes2': TilingSizesVariable,
      'sizes3': TilingSizesVariable,
      'peel': BoolVariable,
      'scalarize_dyn_dims': BoolVariable,
  }

  def transforms(self) -> List[Transform]:
    v = self.assignments
    passes = [
        Tile('matmul_on_tensors', 'linalg.matmul', tile_sizes=v.sizes1),
        Tile('matmul_on_tensors', 'linalg.matmul', tile_sizes=v.sizes2),
        Tile(
            'matmul_on_tensors',
            'linalg.matmul',
            tile_sizes=v.sizes3,
            peel=[0, 1, 2] if v.peel else [])
    ]
    if v.scalarize_dyn_dims:
      passes.append(
          Tile('matmul_on_tensors', 'linalg.matmul', scalarize_dyn_dims=True))
    passes = passes + [
        Vectorize('matmul_on_tensors', 'linalg.matmul'),
        Bufferize(),
        LowerToLLVM(),
    ]
    return passes


class ExpertSparseCompiler(Expert):
  variables = {'options': str}

  def transforms(self) -> List[Transform]:
    v = self.assignments
    self.options = v.options
    return [
        Sparsify(v.options),
    ]


expert_compilerr_1 = ExpertCompiler1(
    sizes1=[256, 256, 256],
    sizes2=[64, 64, 64],
    sizes3=[8, 16, 32],
    pad=True,
    hoist_padding=2)

expert_compilerr_2 = ExpertCompiler2(
    sizes1=[256, 256], sizes2=[8, 16], sizes3=[0, 0, 32])

expert_compilerr_3 = ExpertCompiler3(
    sizes1=[256, 256],
    sizes2=[8, 16, 32],
    sizes3=[8, 32],
    pad=True,
    hoist_padding=3)

expert_compilerr_4 = ExpertCompiler4(
    sizes1=[256, 256, 256],
    sizes2=[64, 64, 64],
    sizes3=[8, 16, 32],
    peel=True,
    scalarize_dyn_dims=True)
