from __future__ import annotations
import functools

from mlir.ir import Module
from mlir.passmanager import PassManager
import typing as tp
from copy import deepcopy

from .variables import Variable

class _TransformThenDescriptor:
  """Python descriptor dispatching `then` on the `Transform` class as either
  class or instance method."""

  # Putting implementations in nested functions here directly instead of methods
  # of Transformation may not be the most elegant approach, but it seems to be
  # the only one that is picked up by the documentation and typecheck tooling.
  def __get__(self, obj: tp.Any, objtype: type = None):
    """Descriptor getter returns either a class-level composition using a
    metaclass or an instance-level composition."""
    # Calling as class method.
    if obj is None:
      assert issubclass(objtype, Transform)
      def then_class(
          other_cls: tp.Union[tp.Type['Transform'],
                              tp.Type['TransformationList']]
      ) -> tp.Type['TransformationList']:
        """Creates a new TransformationList subclass chaining this
        transformation with those provided as argument.

        The new list can be further composed. An instance of the list can be
        created by passing all parameters required by individual
        transformations as keyword arguments to the list constructor.
        """
        assert issubclass(other_cls, (Transform, TransformationList))
        name = objtype.__name__ + "Then" + other_cls.__name__
        transforms = ([objtype, other_cls] if issubclass(other_cls, Transform)
                      else [objtype] + other_cls._transform_classes)
        return TransformListMetaclass(name, (TransformationList,), {},
                                      transforms=transforms)

      return then_class

    # Calling as instance method.
    assert isinstance(obj, Transform)

    def then_instance(
        other: tp.Union['Transform',
                        'TransformationList']) -> 'TransformationList':
      """Creates a new instance of the TransformationList chaining this
      transformation instance with those provided as argument.

      The chained transformations must be instantiated.
      """
      assert isinstance(other, (Transform, TransformationList))
      if isinstance(other, Transform):
        return TransformationList(transforms=[obj, other])

      return TransformationList(transforms=[obj] + other.transforms)

    return then_instance


class Transform:
  """Base class for all parametrized transformations.

  Searchable transformation parameters must be listed in the `variables` field.
  """

  variables: tp.Mapping[str, tp.Union[tp.Type[Transform],
                                      tp.Tuple[tp.Type[Transform],
                                               tp.Any]]] = dict()

  module: Module
  fun_name: str

  def __call__(self, module: Module, fun_name: str):
    self.module = module
    self.fun_name = fun_name
    PassManager.parse(self.pipeline).run(module)
    return module

  def _parse_variables_in_kwargs(self, kwargs: tp.Mapping[str, tp.Any]):
    """Set up instance fields that correspond to known variables from kwargs.

    The values that are missing a default value in `variables` of the current
    class must be provided.
    """
    cls = self.__class__

    for name in kwargs:
      if name not in cls.variables and name not in ('fun_name', 'op_name'):
        raise ValueError(
            f"Unexpected {name} keyword argument, "
            f"{cls.__name__} only accepts {list(cls.variables.keys())}")

    for name in cls.variables:
      if name not in kwargs and not isinstance(cls.variables[name], tuple):
        raise ValueError(f"Missing {name} mandatory keyword argument when "
                         f"constructing {cls}.")
      value = kwargs[name] if name in kwargs else cls.variables[name][1]
      self.__dict__[name] = value

  # Use the Python descriptor mechanism to combine the 'property' mechanism and
  # optional 'classmethod' dispatch. The object is a descriptor that, when read
  # on a class, produces a classmethod-like callable and, when read on an
  # instance, produces an instance callable.
  then = _TransformThenDescriptor()

  def __add__(self, other: tp.Union[Transform, 'TransformationList']) -> 'TransformationList':
    """Create a new transformation list from the current and another
    transformation."""
    return self.then(other)

class Print(Transform):
  """Print intermediate IR.

  Dump the module and do not change it. The transform can be configured as
  follows:
  * `name`: Printer name.
  """

  def __init__(self, name='', **kwargs):
    self.name = name

  def __call__(self, module: Module, fun_name: str):
    print('[[[ IR printer: ' + self.name + ' ]]]')
    module.dump()
    return module

class _TransformListThenDescriptor:
  """Python descriptor dispatching `then` on the `TransformationList` class as
  either class or instance method."""
  def __get__(self, obj: tp.Any, objtype: type):
    if obj is None:
      return objtype._then_cls
    return obj.__add__

class TransformationList:
  """Base class for an Expert compiler that applies transformations in sequence.

  Derived classes that wish to expose their configuration to search need an
  extra `variables` dictionary, which serves as a hook for search. The labels of
  the dictionary need to correspond to the corresponding argument name in the
  derived __init__ function. Such derived classes that chain transformations on
  a single function_name/op_name pair can be constructed using
  `TransformationListMetaclass` that takes care of variables and init kwargs.

  :Parameters:
    - `transforms` - List of transforms to apply in sequence
  """
  transforms: tp.Sequence[Transform]
  _transform_classes: tp.Optional[tp.Sequence[tp.Type[Transform]]] = None
  variables: tp.Mapping[str, tp.Type[Variable]] = dict()

  def __init__(self, transforms: tp.Sequence[Transform]):
    self.transforms = transforms

  def __call__(self, entry_point_name: str, module: Module):
    for transform in self.transforms:
      module = transform(module, entry_point_name)
    return module

  def __add__(
      self, other: tp.Union[Transform,
                            TransformationList]) -> TransformationList:
    """Concatenate two transformation lists.

    The resulting list is no longer suitable for search.
    """
    transforms = [other] if isinstance(other, Transform) else other.transforms
    return TransformationList(transforms=self.transforms + transforms)

  def print_ir(self,
               after_all: bool = False,
               at_begin: bool = False,
               llvm: bool = False) -> TransformationList:
    """Return a new transformation list that prints IR at the given points."""
    transforms = [Print()] if at_begin else []
    for t in self.transforms:
      transforms.append(t)
      if (after_all and 'LowerToLLVM' not in str(t)) or \
         (llvm and 'LowerToLLVM' in str(t)):
        transforms.append(Print(name=str(t)))
    return TransformationList(transforms=transforms)

  then = _TransformListThenDescriptor()


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
      for transform, remapping in zip(transforms, remappings):
        transform_args = {}
        transform_args['fun_name'] = fun_name
        transform_args['op_name'] = op_name
        for name, transform_name in remapping.items():
          if name in kwargs:
            transform_args[transform_name] = kwargs[name]
        self.transforms.append(transform(**transform_args))

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

    attrs['_then_cls'] = then

    return super(TransformListMetaclass, cls).__new__(cls, clsname, bases,
                                                      attrs)

def TransformListFactory(name: str, transforms: tp.Sequence[Transform]):
  """Create a new TransformationList subclss with the given name that performs
  the specified transforms."""
  return TransformListMetaclass(name, (TransformationList,), {},
                                transforms=transforms)
