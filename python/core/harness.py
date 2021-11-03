import sys, time

from collections.abc import Callable
from typing import Any, List, Optional, Sequence, Type, Union

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
  start = time.time()
  run_n_iters(n_iters)
  elapsed_s = time.time() - start
  elapsed_s_per_iter = elapsed_s / n_iters
  gflop_per_s_per_iter = gflop_count / (elapsed_s_per_iter)
  gbyte_per_s_per_iter = gbyte_count / (elapsed_s_per_iter)
  print(f"xxxxxxxxxx : {n_iters} iters time on {1} threads "
        f"in {elapsed_s_per_iter:.{4}}s per iter "
        f"sec ({gflop_per_s_per_iter:.{4}} GFlop/s, "
        f"{gbyte_per_s_per_iter:.{4}} GB/s) "
        f"total time {elapsed_s:.{4}}s ")


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
def compiled_function_element_types_mlir_builder(np_types: Sequence[np.dtype]):
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
      dump_ir_to_file: str = ""):
    assert self.compile_time_problem_sizes_dict is None, \
        f"Problem already compiled, please instantiate a new problem"
    assert_dict_entries_match_keys(compile_time_problem_sizes_dict,
                                   self.problem_sizes_keys)

    self.compile_time_problem_sizes_dict = compile_time_problem_sizes_dict

    with Context() as ctx, Location.unknown() as loc:
      self.mlir_context = ctx
      self.mlir_module = Module.create()
      with InsertionPoint(self.mlir_module.body):
        list_of_sizes = [
            self.compile_time_problem_sizes_dict[t]
            for t in self.problem_sizes_keys
        ]
        types = self.problem_definition.types_mlir_builder(
            *list_of_sizes,
            *compiled_function_element_types_mlir_builder(self.np_types))

        func = self.problem_definition.build_problem_under_context_manager(
            fun_to_benchmark_name, *types)
        wrapper = emit_benchmarking_function(entry_point_name, func)

      def apply_transform_to_entry_point_name(module):
        return transform(entry_point_name, module)

      transformed_module, self.mlir_execution_engine = compile_to_execution_engine(
          self.mlir_module, apply_transform_to_entry_point_name)

      if (len(dump_ir_to_file) > 0):
        f = open(dump_ir_to_file, "w")
        f.write(str(transformed_module))
        f.close()

  def run(self, n_iters: int, entry_point_name: str,
          runtime_problem_sizes_dict: dict):
    assert_dict_entries_match_keys(runtime_problem_sizes_dict,
                                   self.problem_sizes_keys)
    assert_runtime_sizes_compatible_with_compile_time_sizes(
        runtime_problem_sizes_dict, self.compile_time_problem_sizes_dict)

    # 1. Setup NP inputs and outputs
    list_of_sizes = [
        runtime_problem_sizes_dict[t] for t in self.problem_sizes_keys
    ]
    np_input_and_outputs = self.problem_definition.tensors_np_builder(
        *list_of_sizes, *self.np_types)
    # np_input_and_outputs needs to remain live as long as
    # mlir_input_and_outputs_pointers is used
    mlir_input_and_outputs_pointers = get_mlir_abi_compatible_types(
        np_input_and_outputs)

    # 2. Setup function to run, taking just an n_iters arg.
    def run_n_iters(n_iters: int):
      index_ptr_t = ctypes.c_longlong * 1
      self.mlir_execution_engine.invoke(entry_point_name,
                                        *mlir_input_and_outputs_pointers,
                                        index_ptr_t(n_iters))

    # 3. Dry-run.
    run_n_iters(1)

    # 4. Check.
    if self.problem_definition.check_np is not None:
      self.problem_definition.check_np(*np_input_and_outputs)
      # If we checked, do another dry run to warm back up.
      run_n_iters(1)

    # 5. Showtime.
    timed_invoke(
        run_n_iters=run_n_iters,
        gflop_count=self.problem_definition.gflop_count_builder(*list_of_sizes),
        gbyte_count=self.problem_definition.gbyte_count_builder(
            *list_of_sizes, *self.np_types),
        n_iters=n_iters)

    return
