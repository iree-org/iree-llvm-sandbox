#!/usr/bin/env python3

from abc import ABC, abstractmethod
import argparse
import ctypes
from datetime import datetime
import json
import os
import time

import numpy as np
import pandas as pd

from iree.compiler.runtime.np_to_memref import get_ranked_memref_descriptor
from mlir_iterators.dialects import iterators as it
from mlir_iterators.dialects import tabular as tab
from mlir_iterators.dialects import arith, func, memref, scf
from mlir_iterators.execution_engine import ExecutionEngine
from mlir_iterators.ir import (
    Context,  # (Comment preserves formatting.)
    DictAttr,
    IndexType,
    IntegerType,
    InsertionPoint,
    Location,
    MemRefType,
    Module,
    SymbolTable,
    UnitAttr,
)
from mlir_iterators.passmanager import PassManager
from mlir_iterators.runtime.pandas_to_iterators import to_tabular_view_descriptor

_MLIR_RUNNER_UTILS_LIB_ENV = "MLIR_RUNNER_UTILS_LIB"
_MLIR_RUNNER_UTILS_LIB_DEFAULT = "libmlir_runner_utils.so"
_MLIR_C_RUNNER_UTILS_LIB_ENV = "MLIR_C_RUNNER_UTILS_LIB"
_MLIR_C_RUNNER_UTILS_LIB_DEFAULT = "libmlir_c_runner_utils.so"


# Copied from mlir.sandbox.compilation. That package uses the vanilla `mlir`
# package instead of `mlir_iterators` as the rest of this file, so they are
# incompatible.
def emit_benchmarking_function(name: str, bench: func.FuncOp) -> func.FuncOp:
  """Produces the benchmarking function.

  This function calls the given function `bench` as many times as requested by
  its last argument.
  """
  i64_type = IntegerType.get_signless(64)
  nano_time = func.FuncOp("nanoTime", ([], [i64_type]), visibility="private")
  nano_time.attributes["llvm.emit_c_interface"] = UnitAttr.get()

  memref_of_i64_type = MemRefType.get([-1], i64_type)
  wrapper = func.FuncOp(
      # Same signature and an extra buffer of indices to save timings.
      name,
      (bench.arguments.types + [memref_of_i64_type], bench.type.results),
      visibility="public")
  wrapper.attributes["llvm.emit_c_interface"] = UnitAttr.get()

  num_results = len(bench.type.results)
  with InsertionPoint(wrapper.add_entry_block()):
    timer_buffer = wrapper.arguments[-1]
    zero = arith.ConstantOp.create_index(0)
    n_iterations = memref.DimOp(IndexType.get(), timer_buffer, zero)
    one = arith.ConstantOp.create_index(1)
    iter_args = list(wrapper.arguments[-num_results - 1:-1])
    loop = scf.ForOp(zero, n_iterations, one, iter_args)
    with InsertionPoint(loop.body):
      start = func.CallOp(nano_time, [])
      args = list(wrapper.arguments[:-num_results - 1])
      args.extend(loop.inner_iter_args)
      call = func.CallOp(bench, args)
      end = func.CallOp(nano_time, [])
      time = arith.SubIOp(end, start)
      memref.StoreOp(time, timer_buffer, [loop.induction_variable])
      scf.YieldOp(list(call.results))
    func.ReturnOp(loop)

  return wrapper


# Copied from mlir.sandbox.utils. That package uses the vanilla `mlir` package
# instead of `mlir_iterators` as the rest of this file, so they are
# incompatible.
def realign(allocated_unaligned: np.ndarray, byte_alignment: int = 64):
  shape = allocated_unaligned.shape
  dt = allocated_unaligned.dtype
  effective_size_in_bytes = np.prod(shape) * np.dtype(dt).itemsize
  total_size_in_bytes = effective_size_in_bytes + byte_alignment
  buf = np.empty(total_size_in_bytes, dtype=np.byte)
  off = (-buf.ctypes.data % byte_alignment)
  allocated_aligned = buf[off:off +
                          effective_size_in_bytes].view(dt).reshape(shape)
  np.copyto(allocated_aligned, allocated_unaligned)
  assert allocated_aligned.ctypes.data % byte_alignment == 0
  return allocated_aligned


def setup_data(num_elements, dtype):
  """Sets up the input data: two `numpy.arrays` with `num_elements` each of
  type `dtype`."""

  a = realign(np.arange(num_elements, dtype=dtype))
  b = realign(np.arange(num_elements, dtype=dtype))
  return a, b


class Method(ABC):
  """Abstract base class for methods that can be benchmarked.

  Instances of this class will be used with the following protocol during
  benchmarking:

  ```python
  d = m.prepare_inputs(a, b)
  m.compile()
  r = m.run(d)
  ```
  """

  @classmethod
  @property
  def name(cls):
    """Name by which the method can be instantiated."""
    return cls.__name__

  def prepare_inputs(self, a, b):
    """Prepare the inputs according to the method's need. The value returned by
    this function will later be provided to `run`. The default implementation
    returns `(a, b)`."""
    return (a, b)

  def compile(self):
    """Compile the method according to the previous call to `prepare_inputs`,
    if applicable for the method. The default implementation does nothing."""
    pass

  @abstractmethod
  def run_once(self, inputs):
    """Run the benchmark on the previously prepared data once."""
    pass

  def run(self, inputs, num_repetitions):
    """Run the benchmark on the previously prepared data num_repetitions times.
    Implementations may overload this function to run the repetitions in a tight
    loop."""
    run_times_ns = []
    results = []
    for _ in range(num_repetitions):
      start = time.time()
      result = self.run_once(inputs)
      end = time.time()

      run_time_s = end - start
      run_time_ns = int(run_time_s * 10**9)
      run_times_ns.append(run_time_ns)
      results.append(result)

    return run_times_ns, results


class NumpyMethod(Method):
  """Uses `numpy.dot` to compute the inner product."""
  dtype = None

  @classmethod
  @property
  def name(cls):
    return 'numpy'

  def run_once(self, inputs):
    a, b = inputs
    return np.dot(a, b).item()


class IteratorsMethod(Method):

  @classmethod
  @property
  def name(cls):
    return 'iterators'

  def __init__(self):
    super().__init__()
    self.df = None  # Keep reference to prevent GC.
    self.dtype = None
    self.engine = None
    self.sample_input = None

  def prepare_inputs(self, a, b):
    self.df = pd.DataFrame({'a': a, 'b': b}, copy=False)
    self.dtype = self.df.dtypes[0]
    self.sample_input = ctypes.pointer(to_tabular_view_descriptor(self.df[0:0]))
    return ctypes.pointer(to_tabular_view_descriptor(self.df))

  def _load_code(self):
    # Load code from file.
    current_dir = os.path.dirname(os.path.realpath(__file__))
    code_path = os.path.join(current_dir, 'iterators.mlir')
    with open(code_path, 'r') as f:
      code = f.read()

    # Adapt code to dtype.
    type_name = self.dtype.kind + str(self.dtype.itemsize * 8)
    KIND_NAMES = {'i': 'int', 'f': 'float'}
    kind_name = KIND_NAMES[self.dtype.kind]

    code = code\
      .replace('!element_type = i32', '!element_type = ' + type_name) \
      .replace('mapFuncRef = @mul_struct_int', 'mapFuncRef = @mul_struct_' + kind_name) \
      .replace('reduceFuncRef = @sum_int', 'reduceFuncRef = @sum_' + kind_name)

    if self.dtype.kind == 'i':
      code = code.replace('!int_type = i32', '!int_type = ' + type_name)
    else:
      code = code.replace('!float_type = f32', '!float_type = ' + type_name)

    return code

  def compile(self):
    with Context(), Location.unknown():
      it.register_dialect()
      tab.register_dialect()
      code = self._load_code()
      mod = Module.parse(code)
      symbol_table = SymbolTable(mod.operation)
      main_func = symbol_table['main']
      with InsertionPoint(mod.body):
        emit_benchmarking_function('main_bench', main_func)
      pm = PassManager.parse(  # (Comment for better formatting.)
          'convert-iterators-to-llvm,'
          'convert-states-to-llvm,'
          'convert-memref-to-llvm,'
          'convert-scf-to-cf,'
          'convert-func-to-llvm,'
          'reconcile-unrealized-casts,'
          'convert-cf-to-llvm')
    pm.run(mod)
    shared_libs = [
        os.getenv(_MLIR_RUNNER_UTILS_LIB_ENV, _MLIR_RUNNER_UTILS_LIB_DEFAULT),
        os.getenv(_MLIR_C_RUNNER_UTILS_LIB_ENV,
                  _MLIR_C_RUNNER_UTILS_LIB_DEFAULT)
    ]
    self.engine = ExecutionEngine(mod, shared_libs=shared_libs, opt_level=3)

    # Invoke once to move set-up time out of run time.
    self.run(self.sample_input, num_repetitions=1)

  def run_once(self, inputs):
    ctypes_class = np.ctypeslib.as_ctypes_type(self.dtype)
    result = ctypes_class(-1)
    result_ptr = ctypes.pointer(result)
    self.engine.invoke('main', inputs, ctypes.pointer(result_ptr))
    return result.value

  def run(self, inputs, num_repetitions):
    ctypes_class = np.ctypeslib.as_ctypes_type(self.dtype)
    result = ctypes_class(-1)
    result_ptr = ctypes.pointer(result)

    timings = np.empty([num_repetitions], dtype=np.int64)
    timings_desc = get_ranked_memref_descriptor(timings)

    self.engine.invoke('main_bench', inputs, ctypes.pointer(result_ptr),
                       ctypes.pointer(ctypes.pointer(timings_desc)))

    # Assume deterministic results for simplicity.
    results = [result.value] * num_repetitions

    return timings.tolist(), results


# Registry of methods that can be benchmarked.
METHODS = {cls.name: cls for cls in [
    IteratorsMethod,
    NumpyMethod,
]}


def parse_args():
  """Parse the command line arguments using `argparse` and return an
  `argparse.Namespace` object with the bound argument values."""

  parser = argparse.ArgumentParser(
      description='Run benchmark computing inner product of two vectors.')
  parser.add_argument('-r',
                      '--num-repetitions',
                      metavar='N',
                      type=int,
                      default=1,
                      help='Number of repetitions in immediate succession.')
  parser.add_argument('-n',
                      '--num-elements',
                      metavar='N',
                      type=int,
                      default=2**25,
                      help='Number of elements in the two input vectors.')
  parser.add_argument('-t',
                      '--dtype',
                      metavar='T',
                      default='int32',
                      help='Numpy dtype for the elemnts of the input vectors.')
  parser.add_argument('-m',
                      '--method',
                      metavar='M',
                      default='numpy',
                      choices=METHODS.keys(),
                      help='Method to benchmark.')
  return parser.parse_args()


def main():
  # Parse arguments.
  args = parse_args()
  num_elements = args.num_elements
  dtype = np.dtype(args.dtype)
  method = METHODS[args.method]()

  # Set up input data.
  a, b = setup_data(num_elements, dtype)

  # Give method chance to prepare data.
  start = time.time()
  inputs = method.prepare_inputs(a, b)
  end = time.time()
  prepare_time_s = end - start

  # Give method chance to compile code for data.
  start = time.time()
  method.compile()
  end = time.time()
  compile_time_s = end - start

  # Run computation.
  start = time.time()
  run_times_ns, results = method.run(inputs, args.num_repetitions)
  end = time.time()
  total_run_time_s = end - start

  # Assemble and print benchmark data.
  prepare_time_ns = int(prepare_time_s * 10**9)
  compile_time_ns = int(compile_time_s * 10**9)
  total_run_time_ns = int(total_run_time_s * 10**9)

  data = {
      'method': method.name,
      'dtype': dtype.name,
      'num_elements': num_elements,
      'total_run_time_ns': total_run_time_ns,
      'run_times_ns': run_times_ns,
      'prepare_time_ns': prepare_time_ns,
      'compile_time_ns': compile_time_ns,
      'results': results,
      'datetime': datetime.now().isoformat(),
  }

  print(json.dumps(data))


if __name__ == '__main__':
  main()
