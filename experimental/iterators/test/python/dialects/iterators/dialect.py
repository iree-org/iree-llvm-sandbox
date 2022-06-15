# RUN: %PYTHON %s | FileCheck %s

import os

from mlir_iterators.dialects import iterators as it
from mlir_iterators.passmanager import PassManager
from mlir_iterators.execution_engine import ExecutionEngine
from mlir_iterators.ir import Context, Module, IntegerType
import mlir_iterators.all_passes_registration


def run(f):
  print("\nTEST:", f.__name__)
  with Context():
    it.register_dialect()
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
      '%0 = "iterators.constantstream"() {value = []} : () -> (!iterators.stream<!llvm.struct<(i32)>>)'
  )
  # CHECK:      module {
  # CHECK-NEXT:   %0 = "iterators.constantstream"() {value = []} : () -> !iterators.stream<!llvm.struct<(i32)>>
  # CHECK-NEXT: }
  print(mod)


@run
# CHECK-LABEL: TEST: testConvertIteratorsToLlvm
def testConvertIteratorsToLlvm():
  mod = Module.parse('''
      func @main() {
        %0 = "iterators.constantstream"() {value = []} : () -> (!iterators.stream<!llvm.struct<(i32)>>)
        return
      }
      ''')
  pm = PassManager.parse('convert-iterators-to-llvm')
  # Just check that there are no errors...
  pm.run(mod)
  print(mod)


@run
# CHECK-LABEL: TEST: testEndToEnd
def testEndToEnd():
  mod = Module.parse('''
      func @main() attributes { llvm.emit_c_interface } {
        %input = "iterators.constantstream"()
                    { value = [[0 : i32], [1 : i32], [2 : i32], [3 : i32]] } :
                    () -> (!iterators.stream<!llvm.struct<(i32)>>)
        %reduce = "iterators.reduce"(%input) :
                      (!iterators.stream<!llvm.struct<(i32)>>) -> (!iterators.stream<!llvm.struct<(i32)>>)
        "iterators.sink"(%reduce) : (!iterators.stream<!llvm.struct<(i32)>>) -> ()
        return
      }
      ''')
  pm = PassManager.parse('convert-iterators-to-llvm,convert-func-to-llvm,' +
                         'convert-scf-to-cf,convert-cf-to-llvm')
  pm.run(mod)
  engine = ExecutionEngine(mod)
  # CHECK: (6)
  engine.invoke('main')
