# RUN: %PYTHON %s | FileCheck %s

from mlir_iterators.dialects import tuple as tup
from mlir_iterators.ir import (
    Context,
    Location,
    Module,
)


def run(f):
  print("\nTEST:", f.__name__)
  with Context(), Location.unknown():
    tup.register_dialect()
    f()
  return f


# CHECK-LABEL: TEST: testParse
@run
def testParse():
  mod = Module.parse('tuple.from_elements : tuple<>')
  # CHECK:      %tuple = tuple.from_elements  : tuple<>
  print(mod)
