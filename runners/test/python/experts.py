#
# import time

from typing import List
from search import Variable, IntVariable, BoolVariable
from random import choice, randint
from transforms import *


class Assignments:

  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)


class TilingSizesVariable(Variable):
  min_length = 2
  max_length = 3
  valid_sizes = [8, 16, 32, 64, 128, 256]

  def random_value(self):
    result = []
    remaining_sizes = self.valid_sizes
    for x in range(randint(self.min_length, self.max_length)):
      value = choice(remaining_sizes)
      result.append(value)
      remaining_sizes = [s for s in remaining_sizes if s >= value]
    return result


class Expert:

  def __init__(self, **asignments):
    self.assignments = Assignments(**asignments)

  def _pre_transform(self, module, boilerplate_code):

    # TODO: Allow cloning functions from one module to another.
    # Atm we have to resort to string concatenation.
    module = Module.parse(
        str(module.operation.regions[0].blocks[0].operations[0].operation) +
        boilerplate_code)

    return module

  def __call__(self, module, boilerplate_code):
    module = self._pre_transform(module, boilerplate_code)
    for transform in self.transforms():
      transform(module, 'matmul_on_tensors')
    return module

  def transforms(self) -> List[Transform]:
    'Abstract method that returns a list of transforms for given expert.'


class ExpertCompiler1(Expert):
  variables = [
      TilingSizesVariable('sizes1'),
      TilingSizesVariable('sizes2'),
      TilingSizesVariable('sizes3'),
      BoolVariable('pad'),
      IntVariable('hoist_padding', range(2, 4)),
  ]

  def transforms(self) -> List[Transform]:
    v = self.assignments
    return [
        TileAndPad('matmul_on_tensors', 'linalg.matmul', v.sizes1),
        TileAndPad('matmul_on_tensors', 'linalg.matmul', v.sizes2),
        TileAndPad(
            'matmul_on_tensors',
            'linalg.matmul',
            v.sizes3,
            pad=v.pad,
            hoist_padding=v.hoist_padding),
        Vectorize('matmul_on_tensors', 'linalg.matmul'),
        Bufferize(),
        LowerToLLVM(),
    ]


class ExpertCompiler2(Expert):
  variables = [
      TilingSizesVariable('sizes1'),
      TilingSizesVariable('sizes2'),
      TilingSizesVariable('sizes3'),
  ]

  def transforms(self) -> List[Transform]:
    v = self.assignments
    return [
        Fuse('matmul_on_tensors', 'linalg.matmul', v.sizes1),
        Fuse('matmul_on_tensors', 'linalg.matmul', v.sizes2),
        TileAndPad('matmul_on_tensors', 'linalg.matmul', v.sizes3),
        Vectorize('matmul_on_tensors', 'linalg.matmul'),
        Vectorize('matmul_on_tensors', 'linalg.fill'),
        Bufferize(),
        LowerToLLVM(),
    ]


class ExpertCompiler3(Expert):
  variables = [
      TilingSizesVariable('sizes1'),
      TilingSizesVariable('sizes2'),
      TilingSizesVariable('sizes3'),
      BoolVariable('pad'),
      IntVariable('hoist_padding', range(2, 4)),
  ]

  def transforms(self) -> List[Transform]:
    v = self.assignments
    return [
        Fuse('matmul_on_tensors', 'linalg.matmul', v.sizes1),
        TileAndPad(
            'matmul_on_tensors',
            'linalg.matmul',
            v.sizes2,
            pad=v.pad,
            hoist_padding=v.hoist_padding),
        Vectorize('matmul_on_tensors', 'linalg.matmul'),
        TileAndPad('matmul_on_tensors', 'linalg.fill', v.sizes3),
        Vectorize('matmul_on_tensors', 'linalg.fill'),
        Bufferize(),
        LowerToLLVM(),
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
