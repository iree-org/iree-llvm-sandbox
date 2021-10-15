#
# import time

from typing import List, Sequence, Optional

from .search_vars import *
from .transforms import *


# TODO: replace Expert with this
class TransformationList:
  """Base class for an Expert compiler that applies transformations in sequence.

  :Parameters:
    - `transforms` (`List[Transform]`) - List of transforms to apply in sequence
    - `print_ir_after_all` (`bool`) - triggers printing of IR
    - `print_llvmir` (`bool`) - dummy description for y
  """
  transforms = []
  print_ir_after_all = False
  print_llvmir = False

  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

  def __call__(self, entry_point, module):
    for transform in self.transforms:
      is_llvmir = str(transform).find('LowerToLLVM') >= 0
      print_ir = self.print_llvmir if is_llvmir else self.print_ir_after_all

      if print_ir:
        print('[[[ IR after transform: ' + str(transform) + ']]]')
      module = transform(module, entry_point)
      if print_ir:
        print(module)
    return module


# Only for debugging: Print IR after each transform.
print_ir_after_each = False
print_llvmir = False


class Assignments:

  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)


class Expert:

  def __init__(self, **asignments):
    self.assignments = Assignments(**asignments)

  def __call__(self, entry_point, module):
    for transform in self.transforms():
      is_llvmir = str(transform).find('LowerToLLVM') >= 0
      print_ir = print_ir_after_each and (print_llvmir or not is_llvmir)

      if print_ir:
        print('[[[ IR after transform: ' + str(transform) + ']]]')
      module = transform(module, entry_point)
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
class SingleTilingExpert(TransformationList):
  # Variables are the hooks for search, their names must correspond to the
  # __init__
  variables = {
      'sizes': TilingSizesVariable,
      'interchange': InterchangeVariable,
      'peel': PeelingVariable,
      'pad': BoolVariable,
      'pack_padding': PackPaddingVariable,
      'hoist_padding': HoistPaddingVariable,
  }

  def __init__(self, sizes: Sequence[int], interchange: Sequence[int],
               peel: bool, pad: bool, pack_padding: Sequence[int],
               hoist_padding: Sequence[int], **kwargs):
    extra_transforms = [
        Tile(
            'matmul_on_tensors',
            'linalg.matmul',
            tile_sizes=sizes,
            tile_interchange=interchange,
            peel=peel,
            pad=pad,
            pack_padding=pack_padding,
            hoist_padding=hoist_padding),
        Vectorize('matmul_on_tensors', 'linalg.matmul'),
        Bufferize(),
        LowerVectors(),
        LowerToLLVM(),
    ]
    t = extra_transforms if 'transforms' not in kwargs else kwargs[
        'transforms'] + extra_transforms
    d = {'transforms': t}
    kwargs.update(d)
    TransformationList.__init__(self, **kwargs)
