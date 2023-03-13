# RUN: %PYTHON %s | FileCheck %s

import ctypes

import pandas as pd
import numpy as np

from mlir_iterators.runtime.pandas_to_iterators import to_tabular_view_descriptor
from mlir_iterators.dialects import iterators as it
from mlir_iterators.dialects import tabular as tab
from mlir_iterators.dialects import tuple as tup
from mlir_iterators.passmanager import PassManager
from mlir_iterators.execution_engine import ExecutionEngine
from mlir_iterators.ir import Context, Module, IntegerType


def run(f):
  print("\nTEST:", f.__name__)
  with Context():
    it.register_dialect()
    tab.register_dialect()
    tup.register_dialect()
    f()
  return f


# CHECK-LABEL: TEST: testStreamType
@run
def testStreamType():
  i32 = IntegerType.get_signless(32)
  st = it.StreamType.get(i32)
  # CHECK: !iterators.stream<i32>
  print(st)


@run
# CHECK-LABEL: TEST: testParse
def testParse():
  mod = Module.parse(
      '%0 = "iterators.constantstream"() {value = []} : () -> (!iterators.stream<tuple<i32>>)'
  )
  # CHECK:      module {
  # CHECK-NEXT:   %[[V0:.*]] = "iterators.constantstream"() {value = []} : () -> !iterators.stream<tuple<i32>>
  # CHECK-NEXT: }
  print(mod)


@run
# CHECK-LABEL: TEST: testConvertIteratorsToLlvm
def testConvertIteratorsToLlvm():
  mod = Module.parse('''
      func.func @main() {
        %0 = "iterators.constantstream"() {value = []} : () -> (!iterators.stream<tuple<i32>>)
        return
      }
      ''')
  pm = PassManager.parse('builtin.module(convert-iterators-to-llvm)')
  # Just check that there are no errors...
  pm.run(mod.operation)
  print(mod)


@run
# CHECK-LABEL: TEST: testEndToEndStandalone
def testEndToEndStandalone():
  mod = Module.parse('''
      func.func private @sum_tuple(%lhs : tuple<i32>, %rhs : tuple<i32>) -> tuple<i32> {
        %lhsi = tuple.to_elements %lhs : tuple<i32>
        %rhsi = tuple.to_elements %rhs : tuple<i32>
        %i = arith.addi %lhsi, %rhsi : i32
        %result = tuple.from_elements %i : tuple<i32>
        return %result : tuple<i32>
      }
      func.func @main() attributes { llvm.emit_c_interface } {
        %input = "iterators.constantstream"()
                    { value = [[0 : i32], [1 : i32], [2 : i32], [3 : i32]] } :
                    () -> (!iterators.stream<tuple<i32>>)
        %reduce = "iterators.reduce"(%input) {reduceFuncRef = @sum_tuple}
          : (!iterators.stream<tuple<i32>>) -> (!iterators.stream<tuple<i32>>)
        "iterators.sink"(%reduce) : (!iterators.stream<tuple<i32>>) -> ()
        return
      }
      ''')
  pm = PassManager.parse('builtin.module('
                         'convert-iterators-to-llvm,'
                         'decompose-iterator-states,'
                         'decompose-tuples,'
                         'convert-func-to-llvm,'
                         'convert-scf-to-cf,'
                         'convert-cf-to-llvm)')
  pm.run(mod.operation)
  engine = ExecutionEngine(mod)
  # CHECK: (6)
  engine.invoke('main')


@run
# CHECK-LABEL: TEST: testEndToEndWithInput
def testEndToEndWithInput():
  # Set up module that reads data from the outside.
  mod = Module.parse('''
      !struct_type = !llvm.struct<(i32,i64)>
      func.func @main(%input: !tabular.tabular_view<i32,i64>)
          attributes { llvm.emit_c_interface } {
        %stream = iterators.tabular_view_to_stream %input
          to !iterators.stream<!struct_type>
        "iterators.sink"(%stream) : (!iterators.stream<!struct_type>) -> ()
        return
      }
      ''')
  pm = PassManager.parse('builtin.module('
                         'convert-iterators-to-llvm,'
                         'decompose-iterator-states,'
                         'expand-strided-metadata,'
                         'finalize-memref-to-llvm,'
                         'convert-func-to-llvm,'
                         'reconcile-unrealized-casts,'
                         'convert-scf-to-cf,'
                         'convert-cf-to-llvm)')
  pm.run(mod.operation)

  # Set up test data. Note that pandas data frames are have are columnar, i.e.,
  # consist of one memory allocation per column.
  data = {'a': [0, 1, 2], 'b': [3, 4, 5]}
  df = pd.DataFrame.from_dict(data, dtype=int).astype({
      'a': np.int32,  # Corresponds to MLIR's i32.
      'b': np.int64,  # Corresponds to MLIR's i64.
  })
  arg = ctypes.pointer(to_tabular_view_descriptor(df))

  # CHECK:      (0, 3)
  # CHECK-NEXT: (1, 4)
  # CHECK-NEXT: (2, 5)
  engine = ExecutionEngine(mod)
  engine.invoke('main', arg)
