#
# import time

from typing import List
from search import *
from transforms import *

# Only for debugging: Print IR after each transform.
print_ir_after_each = False


class Assignments:

  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)


class Expert:

  def __init__(self, **asignments):
    self.assignments = Assignments(**asignments)

  def _pre_transform(self, module, boilerplate_code):
    benchmark_fun = str(
        module.operation.regions[0].blocks[0].operations[0].operation)
    module = Module.parse(benchmark_fun + boilerplate_code)
    return module

  def __call__(self, module, boilerplate_code):
    module = self._pre_transform(module, boilerplate_code)
    for transform in self.transforms():
      if print_ir_after_each:
        print('[[[ IR after transform: ' + str(transform) + ']]]')
      transform(module, 'matmul_on_tensors')
      if print_ir_after_each:
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
