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

from mlir_iterators.dialects import iterators as it
from mlir_iterators.dialects import tabular as tab
from mlir_iterators.execution_engine import ExecutionEngine
from mlir_iterators.ir import Context, Module
from mlir_iterators.passmanager import PassManager
from mlir_iterators.runtime.pandas_to_iterators import to_tabular_view_descriptor
import mlir_iterators.all_passes_registration


def setup_data(num_elements, dtype):
  """Sets up the input data: two `numpy.arrays` with `num_elements` each of
  type `dtype`."""

  a = np.arange(num_elements, dtype=dtype)
  b = np.arange(num_elements, dtype=dtype)
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
  def run(self, inputs):
    """Run the benchmark on the previously prepared data."""
    pass


class NumpyMethod(Method):
  """Uses `numpy.dot` to compute the inner product."""
  dtype = None

  @classmethod
  @property
  def name(cls):
    return 'numpy'

  def run(self, inputs):
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
    with Context():
      it.register_dialect()
      tab.register_dialect()
      code = self._load_code()
      mod = Module.parse(code)
      pm = PassManager.parse(  # (Comment for better formatting.)
          'convert-iterators-to-llvm,'
          'convert-states-to-llvm,'
          'convert-memref-to-llvm,'
          'convert-func-to-llvm,'
          'reconcile-unrealized-casts,'
          'convert-scf-to-cf,'
          'convert-cf-to-llvm')
    pm.run(mod)
    self.engine = ExecutionEngine(mod, opt_level=3)

    # Invoke once to move set-up time out of run time.
    self.run(self.sample_input)

  def run(self, inputs):
    ctypes_class = np.ctypeslib.as_ctypes_type(self.dtype)
    result = ctypes_class(-1)
    pointer = ctypes.pointer(result)
    self.engine.invoke('main', inputs, ctypes.pointer(pointer))
    return result.value


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
  res = method.run(inputs)
  end = time.time()
  run_time_s = end - start

  # Assemble and print benchmark data.
  prepare_time_us = int(prepare_time_s * 10**6)
  compile_time_us = int(compile_time_s * 10**6)
  run_time_us = int(run_time_s * 10**6)

  data = {
      'method': method.name,
      'dtype': dtype.name,
      'num_elements': num_elements,
      'run_time_us': run_time_us,
      'prepare_time_us': prepare_time_us,
      'compile_time_us': compile_time_us,
      'result': str(res),
      'datetime': datetime.now().isoformat(),
  }

  print(json.dumps(data))


if __name__ == '__main__':
  main()
