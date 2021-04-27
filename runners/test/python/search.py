"""Utilities for search space exploration over linalg operations."""

from mlir.dialects import linalg
from itertools import chain
from random import choice, randrange
from compilation import f16, f32, f64, scalar_types, compile_and_callback
from transforms import expert_compilerr_1


class Variable:
  "Abstract class used as a base for all search variables."

  def assign(self, assignments, value):
    "Assigns variable to a given value in assignments dictionary."

  def random_value(self):
    "Returns a valid random value for this variable."


class TypeVariable(Variable):
  "Linalg operation-specific type variable that defines a scalar component."

  def __init__(self, type_var, scalar_types):
    self.type_var = type_var
    self.scalar_types = scalar_types

  def __repr__(self):
    return f"TypeVariable({self.type_var})"

  def assign(self, assignments, value):
    assignments[self.type_var.name] = value

  def random_value(self):
    return choice(self.scalar_types)


class DimensionVariable(Variable):
  "Linalg operation-specific integer dimension variable."

  def __init__(self, sym, value_range):
    self.sym = sym
    self.value_range = value_range

  def __repr__(self):
    return f"DimensionVariable({self.sym}, {self.value_range})"

  def assign(self, assignments, value):
    assignments[self.sym.symname] = value

  def random_value(self):
    return randrange(self.value_range.start, self.value_range.stop,
                     self.value_range.step)


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


def main():
  op = linalg.matmul
  variables = collect_variables(op, [f32], range(8, 1024 + 1, 8))
  while True:
    assignments = instantiate_variables(variables)
    print(assignments)
    compile_and_callback(op, expert_compilerr_1, lambda x: None, **assignments)


if __name__ == "__main__":
  main()
