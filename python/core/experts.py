from typing import Any, List, NewType, Optional, Sequence, Type

from mlir.ir import *

from .search_vars import *
from .transforms import *


# TODO: replace Expert with this
class TransformationList:
  """Base class for an Expert compiler that applies transformations in sequence.

  Derived classes that whish to expose their configuration to search need an
  extra `variables` dictionary, which serves as a hook for search. The labels of
  the dictionary need to correspond to the corresponding argument name in the
  derived __init__ function.

  :Parameters:
    - `transforms` (`List[Transform]`) - List of transforms to apply in sequence
    - `print_ir_after_all` (`bool`) - triggers printing of IR
    - `print_llvmir` (`bool`) - dummy description for print LLVMIR in particular
  """
  transforms: List
  print_ir_after_all: bool
  print_llvmir: bool

  def __init__(self, **kwargs):
    self.transforms = []
    self.print_ir_after_all = False
    self.print_llvmir = False
    self.__dict__.update(kwargs)

  def __call__(self, entry_point_name: str, module: Module):
    for transform in self.transforms:
      is_llvmir = str(transform).find('LowerToLLVM') >= 0
      print_ir = self.print_llvmir if is_llvmir else self.print_ir_after_all

      if print_ir:
        print('[[[ IR after transform: ' + str(transform) + ']]]')
      module = transform(module, entry_point_name)
      if print_ir:
        print(module)
    return module


def StagedLowerVectorsTransformationList(**kwargs) -> List[LowerVectors]:
  return [
      LowerVectors(stage=0, **kwargs),  # vector.contract
      LowerVectors(stage=1, **kwargs),  # vector.multi_reduction
      LowerVectors(stage=2, **kwargs),  # vector.transfer split
      LowerVectors(stage=3, **kwargs),  # vector.transfer to scf
      LowerVectors(stage=4, **kwargs),  # vector.transfer lowering
      LowerVectors(stage=5, **kwargs),  # vector.shape_cast lowering
      LowerVectors(stage=6, **kwargs),  # vector.transpose lowering
  ]


class LoweringOnlyExpert(TransformationList):
  """Expert compiler that only bufferizes and lowers to LLVM."""

  def __init__(self, transforms: List[Transform] = [], **kwargs):
    post_bufferization_transforms = [] if 'post_bufferization_transforms' \
      not in kwargs else kwargs['post_bufferization_transforms']
    t = transforms + \
        [ Bufferize(**kwargs) ] + \
        post_bufferization_transforms + \
        StagedLowerVectorsTransformationList(**kwargs) + \
        [ LowerToLLVM(**kwargs) ]
    d = {'transforms': t}
    kwargs.update(d)
    TransformationList.__init__(self, **kwargs)


class SingleTilingExpert(TransformationList):
  """Expert compiler that applies a single level of tiling."""

  # Entries in the `variables` dictionary are the hooks for search, their names
  # must correspond to the __init__ function arguments.
  variables = {
      'sizes': TilingSizesVariable,
      'interchange': InterchangeVariable,
      'peel': PeelingVariable,
      'pad': BoolVariable,
      'pack_paddings': PackPaddingVariable,
      'hoist_paddings': HoistPaddingVariable,
  }

  def __init__(self, fun_name: str, op_name: str, sizes: Sequence[int],
               interchange: Sequence[int], peel: Sequence[int], pad: bool,
               pack_paddings: Sequence[int], hoist_paddings: Sequence[int],
               **kwargs):
    extra_transforms = [
        Tile(
            fun_name,
            op_name,
            tile_sizes=sizes,
            tile_interchange=interchange,
            peel=peel,
            pad=pad,
            pack_paddings=pack_paddings,
            hoist_paddings=hoist_paddings,
            **kwargs),
    ]
    if 'vectorize' not in kwargs or kwargs['vectorize']:
      extra_transforms.append(Vectorize(fun_name, op_name, **kwargs))
    extra_transforms.extend(LoweringOnlyExpert(**kwargs).transforms)

    t = extra_transforms if 'transforms' not in kwargs else kwargs[
        'transforms'] + extra_transforms
    d = {
        'sizes': sizes,
        'interchange': interchange,
        'peel': peel,
        'pad': pad,
        'pack_paddings': pack_paddings,
        'hoist_paddings': hoist_paddings,
        'transforms': t
    }
    kwargs.update(d)
    TransformationList.__init__(self, **kwargs)


# Expert compiler that applies two levels of tiling.
# TODO: less copy-pasta, more composing.
class DoubleTilingExpert(TransformationList):
  # Variables are the hooks for search, their names must correspond to the
  # __init__
  variables = {
      'sizes1': TilingSizesVariable,
      'interchange1': InterchangeVariable,
      'peel1': PeelingVariable,
      'pad1': BoolVariable,
      'pack_paddings1': PackPaddingVariable,
      'hoist_paddings1': HoistPaddingVariable,
      'sizes2': TilingSizesVariable,
      'interchange2': InterchangeVariable,
      'peel2': PeelingVariable,
      'pad2': BoolVariable,
      'pack_paddings2': PackPaddingVariable,
      'hoist_paddings2': HoistPaddingVariable,
  }

  def __init__(self, fun_name: str, op_name: str, sizes1: Sequence[int],
               interchange1: Sequence[int], peel1: Sequence[int], pad1: bool,
               pack_paddings1: Sequence[int], hoist_paddings1: Sequence[int],
               sizes2: Sequence[int], interchange2: Sequence[int],
               peel2: Sequence[int], pad2: bool, pack_paddings2: Sequence[int],
               hoist_paddings2: Sequence[int], **kwargs):
    extra_transforms = [
        Tile(
            fun_name,
            op_name,
            tile_sizes=sizes1,
            tile_interchange=interchange1,
            peel=peel1,
            pad=pad1,
            pack_paddings=pack_paddings1,
            hoist_paddings=hoist_paddings1,
            **kwargs),
        Tile(
            fun_name,
            op_name,
            tile_sizes=sizes2,
            tile_interchange=interchange2,
            peel=peel2,
            pad=pad2,
            pack_paddings=pack_paddings2,
            hoist_paddings=hoist_paddings2,
            **kwargs),
    ]
    if 'vectorize' not in kwargs or kwargs['vectorize']:
      extra_transforms.append(Vectorize(fun_name, op_name, **kwargs))
    extra_transforms.extend(LoweringOnlyExpert([], **kwargs).transforms)

    t = extra_transforms if 'transforms' not in kwargs else kwargs[
        'transforms'] + extra_transforms
    d = {
        'sizes1': sizes1,
        'interchange1': interchange1,
        'peel1': peel1,
        'pad1': pad1,
        'pack_paddings1': pack_paddings1,
        'hoist_paddings1': hoist_paddings1,
        'sizes2': sizes2,
        'interchange2': interchange2,
        'peel2': peel2,
        'pad2': pad2,
        'pack_paddings2': pack_paddings2,
        'hoist_paddings2': hoist_paddings2,
        'transforms': t
    }
    kwargs.update(d)
    TransformationList.__init__(self, **kwargs)


# Expert compiler that applies three levels of tiling.
# TODO: less copy-pasta, more composing.
class TripleTilingExpert(TransformationList):
  # Variables are the hooks for search, their names must correspond to the
  # __init__
  variables = {
      'sizes1': TilingSizesVariable,
      'interchange1': InterchangeVariable,
      'peel1': PeelingVariable,
      'pad1': BoolVariable,
      'pack_paddings1': PackPaddingVariable,
      'hoist_paddings1': HoistPaddingVariable,
      'sizes2': TilingSizesVariable,
      'interchange2': InterchangeVariable,
      'peel2': PeelingVariable,
      'pad2': BoolVariable,
      'pack_paddings2': PackPaddingVariable,
      'hoist_paddings2': HoistPaddingVariable,
      'sizes3': TilingSizesVariable,
      'interchange3': InterchangeVariable,
      'peel3': PeelingVariable,
      'pad3': BoolVariable,
      'pack_paddings3': PackPaddingVariable,
      'hoist_paddings3': HoistPaddingVariable,
  }

  def __init__(self, fun_name: str, op_name: str, sizes1: Sequence[int],
               interchange1: Sequence[int], peel1: Sequence[int], pad1: bool,
               pack_paddings1: Sequence[int], hoist_paddings1: Sequence[int],
               sizes2: Sequence[int], interchange2: Sequence[int],
               peel2: Sequence[int], pad2: bool, pack_paddings2: Sequence[int],
               hoist_paddings2: Sequence[int], sizes3: Sequence[int],
               interchange3: Sequence[int], peel3: Sequence[int], pad3: bool,
               pack_paddings3: Sequence[int], hoist_paddings3: Sequence[int],
               **kwargs):
    extra_transforms = [
        Tile(
            fun_name,
            op_name,
            tile_sizes=sizes1,
            tile_interchange=interchange1,
            peel=peel1,
            pad=pad1,
            pack_paddings=pack_paddings1,
            hoist_paddings=hoist_paddings1),
        Tile(
            fun_name,
            op_name,
            tile_sizes=sizes2,
            tile_interchange=interchange2,
            peel=peel2,
            pad=pad2,
            pack_paddings=pack_paddings2,
            hoist_paddings=hoist_paddings2),
        Tile(
            fun_name,
            op_name,
            tile_sizes=sizes3,
            tile_interchange=interchange3,
            peel=peel3,
            pad=pad3,
            pack_paddings=pack_paddings3,
            hoist_paddings=hoist_paddings3),
    ]
    if 'vectorize' not in kwargs or kwargs['vectorize']:
      extra_transforms.append(Vectorize(fun_name, op_name, **kwargs))
    extra_transforms.extend(LoweringOnlyExpert([], **kwargs).transforms)

    t = extra_transforms if 'transforms' not in kwargs else kwargs[
        'transforms'] + extra_transforms
    d = {'transforms': t}
    kwargs.update(d)
    TransformationList.__init__(self, **kwargs)


# Expert compiler that applies the whole sparse compiler.
class ExpertSparseCompiler(TransformationList):
  variables = {'options': str}

  def __init__(self, options):
    self.options = options
    TransformationList.__init__(self, **{'transforms': [Sparsify(options)]})
