import sys

from collections.abc import Callable
from typing import Any, List, Optional, Sequence, Union

import numpy

from mlir.execution_engine import *
from mlir.ir import *
from mlir.runtime import *

from ..core.compilation import compile_to_execution_engine, \
    emit_benchmarking_function
from ..core.problem_definition import *
from ..core.utils import *


# Log everything to stderr and flush so that we have a unified stream to match
# errors/info emitted by MLIR to stderr.
def log(*args):
  print(*args, file=sys.stderr)
  sys.stderr.flush()


def timed_invoke(run_n_iters: Callable, gflop_count: float, gbyte_count: float,
                 n_iters: int):
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
  problem_sizes_keys: Sequence[str]
  compile_time_problem_sizes_dict: dict

  # Result of compilation.
  mlir_context: Any  # TODO: better type
  mlir_module: Any  # TODO: better type
  mlir_execution_engine: Any  # TODO: better type

  def __init__(self, problem_definition: ProblemDefinition,
               problem_sizes_keys: Sequence[str], np_types: Sequence[np.dtype]):
    self.problem_definition = problem_definition
    # Helpers for both compile-time and runtime.
    self.np_types = np_types
    # Information about the problem to enable compilation.
    self.problem_sizes_keys = problem_sizes_keys
    self.compile_time_problem_sizes_dict = None
    # Result of compilation.
    self.mlir_context = None
    self.mlir_module = None
    self.mlir_execution_engine = None

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
    assert_dict_entries_match_keys(compile_time_problem_sizes_dict,
                                   self.problem_sizes_keys)

    self.compile_time_problem_sizes_dict = compile_time_problem_sizes_dict

    with Context() as ctx, Location.unknown() as loc:
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
    assert_dict_entries_match_keys(runtime_problem_sizes_dict,
                                   self.problem_sizes_keys)
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
    timed_invoke(
        run_n_iters=run_n_iters,
        gflop_count=self.problem_definition.gflop_count_builder(
            runtime_problem_sizes_dict),
        gbyte_count=self.problem_definition.gbyte_count_builder(
            runtime_problem_sizes_dict, self.np_types),
        n_iters=n_iters)
