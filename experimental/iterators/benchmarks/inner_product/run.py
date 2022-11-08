#!/usr/bin/env python3

from abc import ABC, abstractmethod
import argparse
from datetime import datetime
import json
import time

import numpy as np


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


# Registry of methods that can be benchmarked.
METHODS = {cls.name: cls for cls in [
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
