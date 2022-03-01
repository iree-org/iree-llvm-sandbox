from collections.abc import Iterable
import itertools
import nevergrad as ng
import typing as tp

from mlir.ir import *
from mlir.passmanager import PassManager
import mlir.dialects.linalg_transform as tx
from mlir.dialects import builtin, pdl

from .variables import *
from .transform import Transform

import mlir.all_passes_registration


class Searchable:
  'Base class for searchable parameters.'

  def __init__(self, name: str):
    self.name = name

  def get_instrumentation(self):
    d = {}
    if self.name != 'searchable_list':
      assert self.name not in d, f'{self.name} key already present'
      d[self.name] = self.instrument()
    return d

  def instrument(self):
    assert False, "instrument implementation must be overridden"

  # Unwrap the np.array from NG's ask() kwargs.
  # This may be overloaded in derived classes to further connect the
  # proposal to concrete values (e.g. permutation, bool etc)
  def extract_from_proposal(self, proposal):
    proposal_slice = proposal.kwargs[self.name]
    if isinstance(proposal_slice, Iterable):
      return [x for x in proposal_slice]
    return proposal_slice


class SearchableList(Searchable):
  """Wrapper class for a list of Searchable. 
  
  Forwards to contained objects."""

  def __init__(self, *searchables):
    Searchable.__init__(self, 'searchable_list')
    self.searchables = list(searchables)

  def get_instrumentation(self):
    d = {}
    for s in self.searchables:
      d = dict(list(d.items()) + list(s.get_instrumentation().items()))
    return d

  def instrument(self):
    instruments = []
    for s in self.searchables:
      instruments = instruments + s.instrument()
    return instruments

  def extract_from_proposal(self, proposal):
    extracts = []
    for s in self.searchables:
      extracts = extracts + s.extract_from_proposal(proposal)
    return extracts


class BoolChoice(Searchable):
  'Searchable that corresponds to a `length` boolean values.'

  length: int = None
  fixed_value: bool = None

  def __init__(self, name: str, length: int = 1):
    Searchable.__init__(self, name)
    self.length = length

  def with_fixed_value(self, fixed_value: bool):
    self.fixed_value = fixed_value
    return self

  def instrument(self):
    if self.fixed_value is not None:
      return ng.p.Choice([1 if self.fixed_value is True else 0],
                         repetitions=self.length)
    return ng.p.Choice([0, 1], repetitions=self.length)

  # Overload to transform the proposal index into an actual permutation
  def extract_from_proposal(self, proposal):
    values = Searchable.extract_from_proposal(self, proposal)
    return [True if x == 1 else False for x in values]


class IntChoice(Searchable):
  'Searchable that corresponds to a `length` integer values.'

  length: int = None
  range: range = None
  values: tp.Sequence[int] = None

  def __init__(self, name: str, length: int = 1):
    Searchable.__init__(self, name)
    self.length = length

  def with_range(self, r):
    assert self.values is None, 'values is set, cannot set range'
    self.range = r
    return self

  def with_values(self, values):
    'with_values can override an existing range (which gets set to None)'
    self.values = values
    self.range = None
    return self

  def instrument(self):
    assert self.range is None or self.values is None, \
      'either range of values must be None'
    if self.range is not None:
      return ng.p.Choice(list(self.range), repetitions=self.length)
    if self.values is not None:
      return ng.p.Choice(self.values, repetitions=self.length)


class Permutation(Searchable):
  'Searchable that corresponds to a single permutation of [0 .. length-1].'
  length: int = None
  permutations: tp.Sequence[tp.Sequence[int]] = None
  # Override the permutations to draw from with a single value
  fixed_permutation: tp.Sequence[int] = None

  def __init__(self, name: str, length: 1):
    Searchable.__init__(self, name)
    self.length = length
    self.permutations = list(itertools.permutations(list(range(self.length))))

  def with_permutation_subset(
      self, fixed_permutation_subset: tp.Sequence[tp.Sequence[int]]):
    for p in fixed_permutation_subset:
      assert len(p) == self.length, 'wrong permutation length'
    self.permutations = fixed_permutation_subset
    return self

  def with_fixed_permutation(self, fixed_permutation: tp.Sequence[int]):
    assert len(fixed_permutation) == self.length, 'wrong permutation length'
    self.fixed_permutation = fixed_permutation
    return self

  def instrument(self):
    if self.fixed_permutation is not None:
      return ng.p.Choice([0])
    return ng.p.Choice(list(range(len(self.permutations))))

  # Overload to transform the proposal index into an actual permutation
  def extract_from_proposal(self, proposal):
    if self.fixed_permutation:
      return self.fixed_permutation
    index = Searchable.extract_from_proposal(self, proposal)
    return self.permutations[index]


class Tile:
  """Interface to tile a linalg op with `tile_sizes`.

  This transform can be configured as follows:
  * `tile_sizes`: Tile sizes used for tiling.
  * `tile_interchange`: Interchange used for tiling.
  * `peel`: Peel the specified loops generated by the tiling pattern. Cannot be
    used together with `pad`.
  """
  tile_sizes: IntChoice = None
  tile_interchange: Permutation = None
  tile_peel: SearchableList = None
  scalarize_dyn_dims: BoolChoice = None

  def __init__(self,
               tile_sizes: SearchableList,
               tile_interchange: Permutation = None,
               tile_peel: SearchableList = None,
               scalarize_dyn_dims: BoolChoice = None):
    self.tile_sizes = tile_sizes
    self.tile_interchange = tile_interchange
    self.tile_peel = tile_peel
    self.scalarize_dyn_dims = scalarize_dyn_dims

  def build_transform_ir_under_insertion_point(self, target: tx.MatchOp,
                                               proposal):
    sizes = self.tile_sizes.extract_from_proposal(proposal)
    interchange = self.tile_interchange.extract_from_proposal(proposal) \
      if self.tile_interchange is not None else [i for i, x in sizes]
    peel = self.tile_peel.extract_from_proposal(proposal) \
      if self.tile_peel is not None else []
    peel_indices = [i for i, x in enumerate(peel) if x is True]
    tiled = tx.TileOp(
        target,
        sizes=sizes,
        interchange=interchange,
        peel=peel_indices,
    )
    # self.scalarize_dyn_dims is a BoolChoice of length 1.
    if self.scalarize_dyn_dims is not None and \
       self.scalarize_dyn_dims.extract_from_proposal(proposal)[0]:
      tiled = tx.TileOp(tiled, scalarize_dyn_dims=True)
    return tiled


def lowering_transform_ir_under_insertion_point():
  tx.BufferizeOp()
  for i in range(7):
    tx.LowerVectorsOp(stages=list(j + 1 for j in range(i + 1)))
  tx.LowerToLLVMOp()
