#
# import time

from typing import List

from .search import *
from .transforms import *

# Only for debugging: Print IR after each transform.
print_ir_after_each = False
print_llvmir = False


class Assignments:

  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)


class Expert:

  def __init__(self, **asignments):
    self.assignments = Assignments(**asignments)

  def _string_stitch_input_ir(self, module, boilerplate_code):
    benchmark_fun = str(
        module.operation.regions[0].blocks[0].operations[0].operation)
    module = Module.parse(benchmark_fun + boilerplate_code)
    return module

  def __call__(self,
               entry_point,
               module,
               boilerplate_code='',
               string_stitch_input_ir=False):
    # TODO: stop hacking strings.
    if string_stitch_input_ir:
      module = self._string_stitch_input_ir(module, boilerplate_code)

    for transform in self.transforms():
      is_llvmir = str(transform).find('LowerToLLVM') >= 0
      print_ir = print_ir_after_each and (print_llvmir or not is_llvmir)

      if print_ir:
        print('[[[ IR after transform: ' + str(transform) + ']]]')
      transform(module, entry_point)
      if print_ir:
        print(module)
    return module

  def transforms(self) -> List[Transform]:
    'Abstract method that returns a list of transforms for given expert.'


class ExpertSparseCompiler(Expert):
  variables = {'options': str}

  def transforms(self) -> List[Transform]:
    v = self.assignments
    self.options = v.options
    return [
        Sparsify(v.options),
    ]


# Expert compiler that applies a single level of tiling.
class SingleTilingExpert(Expert):
  variables = {
      'sizes1': TilingSizesVariable,
      'interchange': InterchangeVariable,
      'pad': PaddingVariable,
      'peel': PeelingVariable,
      'hoist_padding': HoistPaddingVariable,
  }

  def transforms(self) -> List[Transform]:
    v = self.assignments
    return [
        Tile(
            'matmul_on_tensors',
            'linalg.matmul',
            tile_sizes=v.sizes1,
            tile_interchange=v.interchange,
            pad=v.pad,
            peel=v.peel,
            hoist_padding=v.hoist_padding),
        Vectorize('matmul_on_tensors', 'linalg.matmul'),
        Bufferize(),
        LowerVectors(),
        LowerToLLVM(),
    ]
