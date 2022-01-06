import argparse
import re
import sys
import os
import time
from collections import defaultdict

from typing import AbstractSet, Any, Callable, List, Mapping, Optional, Sequence, Union

import numpy

from mlir.execution_engine import *
from mlir.ir import *
from mlir.runtime import *
from mlir.iree_sandbox import register_sandbox_passes_and_dialects

from ..core.compilation import compile_to_execution_engine, \
    emit_benchmarking_function
from ..core.experts import TransformationList
from ..core.problem_definition import *
from ..core.utils import *


# Log everything to stderr and flush so that we have a unified stream to match
# errors/info emitted by MLIR to stderr.
def log(*args):
  print(*args, file=sys.stderr)
  sys.stderr.flush()


TimingResults = Mapping[str, Sequence[float]]


def timed_invoke(run_n_iters: Callable, gflop_count: float, gbyte_count: float,
                 n_iters: int) -> TimingResults:
  elapsed_ns = run_n_iters(n_iters)
  elapsed_s = np.flip(np.sort(elapsed_ns / 1.e9))
  elapsed_s_per_iter = [                  \
      elapsed_s[0],                       \
      elapsed_s[((n_iters *  1) // 100)], \
      elapsed_s[((n_iters * 10) // 100)], \
      elapsed_s[((n_iters * 25) // 100)], \
      elapsed_s[((n_iters * 50) // 100)], \
      elapsed_s[((n_iters * 75) // 100)], \
      elapsed_s[((n_iters * 90) // 100)], \
      elapsed_s[((n_iters * 99) // 100)], \
      elapsed_s[-1]                       \
  ]
  gbyte_per_s_per_iter = [(gbyte_count / sec) for sec in elapsed_s_per_iter]
  gflop_per_s_per_iter = [(gflop_count / sec) for sec in elapsed_s_per_iter]
  print(f'xxxxxxxxxx : {n_iters} iters time on {1} threads')
  line = '-' * 120
  header_data = \
      ['slowest', 'p1', 'p10', 'p25', 'p50', 'p75', 'p90', 'p99', 'fastest']
  data = [ \
      header_data +              ['unit'], \
      elapsed_s_per_iter +    ['seconds'], \
      gflop_per_s_per_iter + ['GFlops/s'], \
      gbyte_per_s_per_iter +    ['GBs/s']  \
  ]
  print(line)
  format_str = '{:>12s}' * len(data[0])
  print(format_str.format(*data[0]))
  print(line)
  format_str = '{:>12.1e}' * (len(data[0]) - 1) + '{:>12s}'
  print(format_str.format(*data[1]))
  for i in range(2, len(data)):
    format_str = '{:>12.2f}' * (len(data[0]) - 1) + '{:>12s}'
    print(format_str.format(*data[i]))

  return {
      "elapsed_s_per_iter": elapsed_s_per_iter,
      "gbyte_per_s_per_iter": gbyte_per_s_per_iter,
      "gflop_per_s_per_iter": gflop_per_s_per_iter,
  }


# TODO: support more than just RankedTensorType.
def get_mlir_abi_compatible_type(value):
  return get_ranked_memref_descriptor(value)


# TODO: tighter type than Sequence[Any].
def get_mlir_abi_compatible_types(input_and_outputs: Sequence[Any]):
  # Arguments must be passed as pointers.
  return [
      ctypes.pointer(ctypes.pointer(get_mlir_abi_compatible_type(t)))
      for t in input_and_outputs
  ]


# Return the list of mlir types matching np_types 1-to-1.
def compiled_function_element_types_mlir_builder(
    np_types: Sequence[np.dtype]) -> List[Type]:
  return [np_type_to_mlir_type(t) for t in np_types]


class ProblemInstance:
  problem_definition: ProblemDefinition

  # Helpers for both compile-time and runtime.
  np_types: Sequence[np.dtype]

  # Information about the problem to enable compilation.
  compile_time_problem_sizes_dict: dict

  # Result of compilation.
  mlir_context: Any  # TODO: better type
  mlir_module: Any  # TODO: better type
  mlir_execution_engine: Any  # TODO: better type

  def __init__(self, problem_definition: ProblemDefinition,
               np_types: Sequence[np.dtype]):
    self.problem_definition = problem_definition
    # Helpers for both compile-time and runtime.
    self.np_types = np_types
    # Information about the problem to enable compilation.
    self.compile_time_problem_sizes_dict = None
    # Result of compilation.
    self.mlir_context = None
    self.mlir_module = None
    self.mlir_execution_engine = None

  def __assert_matching_mapping_keys(self, mapping: Mapping[str, Any]):
    if not hasattr(self.problem_definition, 'keys'):
      return
    assert_dict_entries_match_keys(mapping, self.problem_definition.keys)

  def compile(
      self,
      entry_point_name: str,
      fun_to_benchmark_name: str,
      compile_time_problem_sizes_dict: dict,
      # TODO: Better type than Callable.
      transform: Callable,
      dump_ir_to_file: str = ''):
    assert self.compile_time_problem_sizes_dict is None, \
        f'Problem already compiled, please instantiate a new problem'
    self.__assert_matching_mapping_keys(compile_time_problem_sizes_dict)

    self.compile_time_problem_sizes_dict = compile_time_problem_sizes_dict

    with Context() as ctx, Location.unknown() as loc:
      register_sandbox_passes_and_dialects(ctx)
      self.mlir_context = ctx
      self.mlir_module = Module.create()
      with InsertionPoint(self.mlir_module.body):
        types = self.problem_definition.types_mlir_builder(
            self.compile_time_problem_sizes_dict,
            compiled_function_element_types_mlir_builder(self.np_types))

        func = self.problem_definition.build_problem_under_context_manager(
            fun_to_benchmark_name, types)
        wrapper = emit_benchmarking_function(entry_point_name, func)

      def apply_transform_to_entry_point_name(module):
        return transform(entry_point_name, module)

      transformed_module, self.mlir_execution_engine = compile_to_execution_engine(
          self.mlir_module, apply_transform_to_entry_point_name)

      if (len(dump_ir_to_file) > 0):
        f = open(dump_ir_to_file, 'w')
        f.write(str(transformed_module))
        f.close()

  def run(self,
          n_iters: int,
          entry_point_name: str,
          runtime_problem_sizes_dict: dict,
          dump_obj_to_file: str = ''):
    self.__assert_matching_mapping_keys(runtime_problem_sizes_dict)
    assert_runtime_sizes_compatible_with_compile_time_sizes(
        runtime_problem_sizes_dict, self.compile_time_problem_sizes_dict)

    # 1. Setup NP inputs and outputs
    np_input_and_outputs = self.problem_definition.tensors_np_builder(
        runtime_problem_sizes_dict, self.np_types)
    # np_input_and_outputs needs to remain live as long as
    # np_input_and_outputs_pointers is used
    np_input_and_outputs_pointers = get_mlir_abi_compatible_types(
        np_input_and_outputs)

    # 2. Setup function to run, taking a np array of .
    def run_n_iters(n_iters: int):
      np_timers = np.zeros([n_iters], dtype=np.int64)
      np_timers_pointer = get_mlir_abi_compatible_types([np_timers]).pop()
      self.mlir_execution_engine.invoke(entry_point_name,
                                        *np_input_and_outputs_pointers,
                                        np_timers_pointer)
      return np_timers

    # 3. Dry-run.
    run_n_iters(1)

    # Now dump to obj file as the JIT compilation actually happened.
    if (len(dump_obj_to_file) > 0):
      self.mlir_execution_engine.dump_to_object_file(dump_obj_to_file)

    # 4. Check.
    # TODO: this checks seems to be always true as `check_np` is a function
    # defined to be just `pass` at the base class level, nobody overrides it as
    # attribute to be None.
    if self.problem_definition.check_np is not None:
      self.problem_definition.check_np(*np_input_and_outputs)
      # If we checked, do another dry run to warm back up.
      run_n_iters(1)

    # 5. Showtime.
    return timed_invoke(run_n_iters=run_n_iters,
                        gflop_count=self.problem_definition.gflop_count_builder(
                            runtime_problem_sizes_dict),
                        gbyte_count=self.problem_definition.gbyte_count_builder(
                            runtime_problem_sizes_dict, self.np_types),
                        n_iters=n_iters)


def _pytimed(callback: Callable[..., None], *args: Any, **kwargs: Any):
  """Call the given callback and return time in nanoseconds as result."""
  start_time = time.monotonic_ns()
  results = callback(*args, **kwargs)
  end_time = time.monotonic_ns()
  duration = (end_time - start_time)
  return duration


def _run_benchmark_n_iters(callback: Callable[[int], None], n_iters: int,
                           *args: Any):
  """Call the given callback `n_iters` times and return the times as a 1-d array."""
  return np.asarray([_pytimed(callback, *args) for _ in range(n_iters)])


def _parse_problem_sizes(argument: str) -> Sequence[Union[int, Sequence[int]]]:
  """Parse a problem size argument into a possibly nested integer sequence.

  Examples:
  64,128 -> [64, 128]
  32,32,[1,1] -> [32, 32, [1, 1]]
  """
  problem_sizes = []
  while argument:
    # Match size.
    match = re.match(r"""[,]?\d+""", argument)
    if match:
      problem_sizes.append(int(match.group().lstrip(',')))
      argument = argument[match.end():]
      continue
    # Match nested sizes.
    match = re.match(r"""[,]?\[[0-9,]+\]""", argument)
    if match:
      nested = match.group().lstrip(',')[1:-1]
      problem_sizes.append([int(elem) for elem in nested.split(',')])
      argument = argument[match.end():]
      continue
    raise ValueError()
  return problem_sizes


def _parse_read_only(argument: str) -> Sequence[str]:
  """Parse a sequence of read only dimensions.

  Examples:
  k,m -> ['k', 'm']
  [] -> []
  """
  if argument == '[]':
    return []
  return argument.split(',')


def test_argparser(benchmark_name: str,
                   default_problem_sizes_list: Sequence[Sequence[int]],
                   default_expert_list: Sequence[int],
                   default_runtime_only_list: Sequence[Sequence[str]],
                   default_spec_list: Sequence[str]) -> argparse.Namespace:
  """Test argument parser.

  Creates an argument parser and returns the parsed arguments.

  Arguments:
  benchmark_name: Benchmark name.
  default_problem_sizes_list: Default problem sizes.
  default_expert_list: Default expert indices.
  default_runtime_only_list: Default runtime only dimensions list.
  default_spec_list: Default specification list.
  """
  parser = argparse.ArgumentParser(description=benchmark_name)
  parser.add_argument('--problem_sizes_list', '-p',
                      type=_parse_problem_sizes,
                      nargs='+',
                      help='problem sizes (e.g., -p 32,32,64 8,8,8)',
                      default=default_problem_sizes_list)
  parser.add_argument('--expert_list', '-e', type=int, nargs='+',
                      help='experts (e.g., -e 0 1 2)',
                      default=default_expert_list)
  parser.add_argument('--runtime_only_list', '-r',
                      type=_parse_read_only,
                      nargs='+',
                      help='runtime only dimensions (e.g., -r k,m k [])',
                      default=default_runtime_only_list)
  parser.add_argument('--spec_list', '-s',
                      type=str,
                      nargs='+',
                      help='problem specifications (e.g., -s mk,kn km,kn)',
                      default=default_spec_list)
  return parser.parse_args(sys.argv[1:])


def test_harness(problem_factory: Callable[
    [Mapping[str, Any], Sequence[np.dtype]], ProblemDefinition],
                 np_types_list: Sequence[Sequence[np.dtype]],
                 problem_sizes_list: Sequence[Mapping[str, Any]],
                 experts: Sequence[TransformationList],
                 n_iters: int = 1,
                 function_name: str = 'tested_function',
                 runtime_only_sizes: AbstractSet[str] = set(),
                 **kwargs) -> Mapping[str, TimingResults]:
  """Test runner facility.

  Compiles and runs the a test or a benchmark for a cross-product of possible
  argument types, problem sizes and compilation experts. Collects and prints the
  results to the standard output.

  Arguments:
  problem_factory: A callable to construct a ProblemDefinition given the size
    mapping and the argument type choice. May be called multiple times.
  np_type_list: A list of elemental type lists to try (each inner list must have
    as many entries as the problem has arguments).
  problem_sizes_list: A list of size mappings to try.
  experts: A list of compilation experts to try.
  n_iters: Number of times to run the test.
  function_name: Name of the function in which the IR is emitted, this name can
   be used by compilation experts to target the transformation.
  runtime_only_sizes: A set of size keys that should be treated as unknown (-1)
    at compilation time and only set at runtime.

  Keyword arguments:
  numpy_benchmark: A callable accepting a list of NumPy tensors, the current
    size mapping and the type selection that performs the computation using
    Numpy. If the `BENCHMARK_NUMPY` environment variable is set and the argument
    is provided, it will be called `n_iters` times for the purpose of measuring
    baseline performance.
  pytorch_benchmark: A callable accepting a list of PyTorch tensors, the current
    size mapping and the type selection that performs the computation using
    PyTorch. If the `BENCHMARK_TORCH` environment variable is set and the
    argument is provided, it will be called `n_iters` times for the purpose of
    measuring baseline performance.

  Returns: A dictionary of all collected benchmark results.
  """

  results = {}

  for np_types in np_types_list:
    for problem_sizes_dict in problem_sizes_list:
      compile_time_problem_sizes_dict = {
          key: (value if key not in runtime_only_sizes else -1)
          for key, value in problem_sizes_dict.items()
      }
      runtime_problem_sizes_dict = problem_sizes_dict

      # Init printing.
      print(
          f'\n###############################################################\n'
          f'Compile-time problem size {compile_time_problem_sizes_dict}\n'
          f'Runtime problem size {runtime_problem_sizes_dict}\n'
          f'Problem types {np_types}')
      for expert in experts:
        print(f'\nCompilation expert {expert}')
        problem_definition = problem_factory(problem_sizes_dict, np_types)
        problem = ProblemInstance(problem_definition, np_types)

        problem.compile(
            entry_point_name='main',
            fun_to_benchmark_name=function_name,
            compile_time_problem_sizes_dict=compile_time_problem_sizes_dict,
            transform=expert,
            dump_ir_to_file=kwargs.get('dump_ir_to_file', ''))

        results[str(expert)] = problem.run(
            n_iters=n_iters,
            entry_point_name='main',
            runtime_problem_sizes_dict=runtime_problem_sizes_dict,
            dump_obj_to_file=kwargs.get('dump_obj_to_file', ''))

      problem_definition = problem_factory(problem_sizes_dict, np_types)
      gflops = problem_definition.gflop_count_builder(problem_sizes_dict)
      gbytes = problem_definition.gbyte_count_builder(problem_sizes_dict,
                                                      np_types)

      if 'numpy_benchmark' in kwargs and os.environ.get('BENCHMARK_NUMPY'):
        print('\nNumPy reference\n')
        args = problem_definition.tensors_np_builder(problem_sizes_dict,
                                                     np_types)
        results['numpy'] = timed_invoke(
            lambda n: _run_benchmark_n_iters(kwargs['numpy_benchmark'], n, args,
                                             problem_sizes_dict, np_types),
            gflops, gbytes, n_iters)

      if 'pytorch_benchmark' in kwargs and os.environ.get('BENCHMARK_TORCH'):
        print('\nPyTorch reference\n')
        import torch
        torch.set_num_threads(1)
        numpy_args = problem_definition.tensors_np_builder(
            problem_sizes_dict, np_types)
        args = list(map(torch.from_numpy, numpy_args))
        results['pytorch'] = timed_invoke(
            lambda n: _run_benchmark_n_iters(kwargs[
                'pytorch_benchmark'], n, args, problem_sizes_dict, np_types),
            gflops, gbytes, n_iters)

    return results
