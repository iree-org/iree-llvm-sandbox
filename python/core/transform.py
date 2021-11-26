from __future__ import annotations

from mlir.ir import Module
from mlir.passmanager import PassManager
import typing as tp
from copy import deepcopy

from .search_vars import Variable


class Transform:
  """Base class for all parametrized transformations.
  
  Searchable transformation parameters must be listed in the `variables` field.
  """

  variables: tp.Mapping[str, tp.Type[Variable]] = dict()

  module: Module
  fun_name: str

  def __call__(self, module: Module, fun_name: str):
    self.module = module
    self.fun_name = fun_name
    PassManager.parse(self.pipeline).run(module)
    return module

  def _parse_variables_in_kwargs(self,
                                 kwargs: tp.Mapping[str, tp.Any],
                                 defaults: tp.Mapping[str, tp.Any] = dict()):
    """Set up instance fields that correspond to known variables from kwargs.
    
    Use the values if `defaults` if `kwargs` does not have one for the given
    variable. Either `kwargs` or `defaults` must contain a value for all known
    variables. 
    """
    cls = self.__class__
    for name in cls.variables:
      if name not in kwargs and name not in defaults:
        raise ValueError(f"Missing {name} keyword argument when constructing "
                         f"{cls} with no default provided.")
      value = kwargs[name] if name in kwargs else defaults[name]
      self.__dict__[name] = value

  @classmethod
  def then(cls, other_cls: tp.Union[tp.Type[Transform],
                                    tp.Type['TransformationList']]):
    name = cls.__name__ + "Then" + other_cls.__name__
    transforms = [cls, other_cls] if issubclass(other_cls, Transform) \
                 else [cls] + other_cls._transform_classes
    return TransformListMetaclass(name, (TransformationList,), {},
                                  transforms=transforms)


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
  transforms: tp.List
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


def _get_name_remapping(transform_classes: tp.Sequence[tp.Type[Transform]]):
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

  def __new__(cls, clsname: str, bases: tp.Tuple[tp.Type],
              attrs: tp.Mapping[str, tp.Any],
              transforms: tp.Sequence[tp.Type[Transform]]):
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
    def then(cls, other_cls: tp.Union[tp.Type[Transform],
                                      tp.Type[TransformationList]]):
      name = cls.__name__ + "Then" + other_cls.__name__
      transforms = cls._transform_classes + [other_cls] \
                   if issubclass(other_cls, Transform) \
                   else cls._transform_classes + other_cls._transform_classes
      return TransformListMetaclass(name, (TransformationList,), {},
                                    transforms=transforms)

    attrs['then'] = then

    return super(TransformListMetaclass, cls).__new__(cls, clsname, bases,
                                                      attrs)


def TransformListFactory(name: str, transforms: tp.Sequence[Transform]):
  """Create a new TransformationList subclss with the given name that performs
  the specified transforms."""
  return TransformListMetaclass(name, (TransformationList,), {},
                                transforms=transforms)
