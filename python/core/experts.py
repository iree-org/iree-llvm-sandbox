from typing import Any, List, NewType, Optional, Sequence
# Qualified import to disambiguate with mlir.ir.Type.
__import__('typing', fromlist=['Type'])
from copy import deepcopy

from mlir.ir import *

from .search_vars import *
from .transforms import *


# TODO: replace Expert with this
class TransformationList:
  """Base class for an Expert compiler that applies transformations in sequence.

  Derived classes that whish to expose their configuration to search need an
  extra `variables` dictionary, which serves as a hook for search. The labels of
  the dictionary need to correspond to the corresponding argument name in the
  derived __init__ function. Such derived classes that chain transformations on
  a single function_name/op_name pair can be constructed using
  `TransformationListMetaclass` that takes care of variables and init kwargs.

  :Parameters:
    - `transforms` (`List[Transform]`) - List of transforms to apply in sequence
    - `print_ir_after_all` (`bool`) - triggers printing of IR
    - `print_llvmir` (`bool`) - dummy description for print LLVMIR in particular
  """
  transforms: List
  print_ir_at_begin: bool
  print_ir_after_all: bool
  print_llvmir: bool

  def __init__(self, **kwargs):
    self.transforms = []
    self.print_ir_at_begin = False
    self.print_ir_after_all = False
    self.print_llvmir = False
    self.__dict__.update(kwargs)

  def __call__(self, entry_point_name: str, module: Module):
    if self.print_ir_at_begin:
      print(module)
    for transform in self.transforms:
      is_llvmir = str(transform).find('LowerToLLVM') >= 0
      print_ir = self.print_llvmir if is_llvmir else self.print_ir_after_all

      if print_ir:
        print('[[[ IR after transform: ' + str(transform) + ']]]')
      module = transform(module, entry_point_name)
      if print_ir:
        print(module)
    return module


def _get_name_remapping(transform_classes: Sequence[typing.Type[Transform]]):
  """Given a list of Transform classes, create a list of mappings from unique
  suffixed names back to per-transformation variable names."""
  seen_names = dict()
  for transform in transform_classes:
    for name in transform.variables:
      if name not in seen_names:
        seen_names[name] = 0
      seen_names[name] += 1
  next_index = {name: 1 for name in seen_names}
  remappings = []
  for transform in transform_classes:
    remapping = dict()
    for name in transform.variables:
      expert_name = name
      if seen_names[name] > 1:
        expert_name += str(next_index[name])
        next_index[name] += 1
      remapping[expert_name] = name
    remappings.append(remapping)
  return remappings


class TransformListMetaclass(type):
  """Metaclass for TransformationList subclasses that chain transformations.
  
  Given the list of Transformation subclasses as `transforms` kwarg, creates a
  new class that derives TransformationList and instantiates the transformations
  in its constructor. The new subclass has the `variables` field suitable for
  search defined by combining the `variables` fields of each transformation. If
  transformations have identical variable names, they will be suffixed with
  increasing integer numbers, e.g., two transformations having the "sizes"
  variable will result in a list with "sizes1" and "sizes2" variables, regardless
  of the position of these transformations in the list. The constructor of the
  new class accepts kwargs with the same names as `variables`.

  Classes created with this metaclass can be in turn chained using the `then`
  class method, which produces a new class following the same rules.
  """

  def __new__(cls, clsname, bases, attrs, transforms):
    if 'variables' in attrs and transforms:
      raise ValueError(
          "TransformList metaclass would override the list of variables.")

    remappings = _get_name_remapping(transforms)
    variables = dict()
    for transform, remapping in zip(transforms, remappings):
      variables.update(
          {name: transform.variables[remapping[name]] for name in remapping})
    attrs['variables'] = variables

    def init(self, fun_name: str, op_name: str, **kwargs):
      self.transforms = []
      kwargs['fun_name'] = fun_name
      kwargs['op_name'] = op_name
      if 'print_ir_at_begin' not in kwargs:
        kwargs['print_ir_at_begin'] = False
      if 'print_ir_after_all' not in kwargs:
        kwargs['print_ir_after_all'] = False
      if 'print_llvmir' not in kwargs:
        kwargs['print_llvmir'] = False
      for transform, remapping in zip(transforms, remappings):
        transform_args = deepcopy(kwargs)
        for name, transform_name in remapping.items():
          if transform_name == name:
            continue
          transform_args[transform_name] = transform_args[name]
          del transform_args[name]
        self.transforms.append(transform(**transform_args))
      self.__dict__.update(kwargs)

    attrs['__init__'] = init
    attrs['_transform_classes'] = transforms

    @classmethod
    def then(cls, other_cls):
      return TransformListMetaclass(
          cls.__name__ + "Then" + other_cls.__name__, (TransformationList,), {},
          transforms=cls._transform_classes + other_cls._transform_classes)

    attrs['then'] = then

    return super(TransformListMetaclass, cls).__new__(cls, clsname, bases,
                                                      attrs)


def LowerVectorFactory(stage):
  """Create a new Transformation class that binds the lowering stage to the
  given number at class construction time rather than at object construction."""

  def init(self, **kwargs):
    LowerVectors.__init__(self, stage, **kwargs)

  return type('LowerVectors' + str(stage), (LowerVectors,), {'__init__': init})


def TransformListFactory(name: str, transforms: Sequence[Transform]):
  """Create a new TransformationList subclss with the given name that performs
  the specified transforms."""
  return TransformListMetaclass(name, (TransformationList,), {},
                                transforms=transforms)


# TODO: This is still boilerplaty, allow chaining transformations directly to form a list.
Bufferization = TransformListFactory('Bufferization', [Bufferize])
Decomposition = TransformListFactory('Decomposition',
                                     [DecomposeToLowerDimensionalNamedOp])
Generalization = TransformListFactory('Generalization', [Generalize])
LLVMLowering = TransformListFactory('LLVMLowering', [LowerToLLVM])
Tiling = TransformListFactory('Tiling', [Tile])
Vectorization = TransformListFactory('Vectorization', [Vectorize])
VectorLowering = TransformListFactory('VectorLowering',
                                      [LowerVectorFactory(i) for i in range(7)])

# TODO: After DecomposeToLowerDimensionalNamedOp the op_name to anchor on
# changes: we need a better control mechanism.
LoweringOnlyExpert = Bufferization.then(VectorLowering).then(LLVMLowering)
SingleTilingExpert = Tiling.then(Decomposition).then(Vectorization).then(
    LoweringOnlyExpert)
DoubleTilingExpert = Tiling.then(SingleTilingExpert)
TripleTilingExpert = Tiling.then(DoubleTilingExpert)


# Expert compiler that applies the whole sparse compiler.
class ExpertSparseCompiler(TransformationList):
  variables = {'options': str}

  def __init__(self, options):
    self.options = options
    TransformationList.__init__(self, **{'transforms': [Sparsify(options)]})
