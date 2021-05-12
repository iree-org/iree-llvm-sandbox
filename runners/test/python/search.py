"""Utilities for search space exploration over linalg operations."""

from itertools import chain
from random import choice, randrange, randint


class Variable:
  "Abstract class used as a base for all search variables."

  def __init__(self, name):
    self.name = name

  def assign(self, assignments, value):
    "Assigns variable to a given value in assignments dictionary."

    assignments[self.name] = value

  def random_value(self):
    "Abstract method that returns a valid random value for this variable."


class TypeVariable(Variable):
  "Linalg operation-specific type variable that defines a scalar component."

  def __init__(self, type_var, scalar_types):
    Variable.__init__(self, type_var.name)
    self.type_var = type_var
    self.scalar_types = scalar_types

  def __repr__(self):
    return f"TypeVariable({self.type_var})"

  def random_value(self):
    return choice(self.scalar_types)


class IntVariable(Variable):
  "Linalg operation-specific integer dimension variable."

  def __init__(self, name, value_range):
    Variable.__init__(self, name)
    self.value_range = value_range

  def __repr__(self):
    return f"IntVariable({self.name}, {self.value_range})"

  def random_value(self):
    return randrange(self.value_range.start, self.value_range.stop,
                     self.value_range.step)


class BoolVariable(Variable):
  "Boolean flag variable."

  def __repr__(self):
    return f"BoolVariable({self.name})"

  def random_value(self):
    return randint(0, 1) == 0


class DimensionVariable(IntVariable):
  "Variable that corresponds to the operation dimensions."

  def __init__(self, sym, value_range):
    IntVariable.__init__(self, sym.symname, value_range)
    self.sym = sym

  def __repr__(self):
    return f"DimensionVariable({self.sym}, {self.value_range})"


def collect_variables(op, scalar_types, dimension_value_range):
  type_vars = set()
  syms = set()
  for tdef in chain(op.model.inputs, op.model.outputs):
    type_vars.add(tdef.type_var)
    for sym in tdef.shape:
      syms.add(sym)

  variables = []
  for type_var in type_vars:
    variables.append(TypeVariable(type_var, scalar_types))
  for sym in syms:
    variables.append(DimensionVariable(sym, dimension_value_range))

  return variables


def instantiate_variables(variables):
  assignments = {}
  for variable in variables:
    variable.assign(assignments, variable.random_value())
  return assignments
