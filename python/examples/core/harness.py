import argparse
import re
import sys
import os
import time
from collections import defaultdict

from typing import AbstractSet, Any, Callable, List, Mapping, Optional, Sequence, Union

import numpy
import pandas
import seaborn
import matplotlib

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
ProblemSizes = Sequence[Union[int, Sequence[int]]]


class Measurements:
  """Class storing measurement configuration and results in data frame."""
  config_keys = [
    "expert",
    "np_types",
    "dynamic_at_compile_time",
    "runtime_problem_sizes_dict"
  ]
  data_keys = [
    "elapsed_s_per_iter",
    "gbyte_per_s_per_iter",
    "gflop_per_s_per_iter"
  ]

  def __init__(self):
    self.data = pandas.DataFrame(
        dict([(col, []) for col in self.config_keys + self.data_keys]))

  def append(self, expert: str, np_types: Sequence[np.dtype],
             dynamic_at_compile_time_sizes: AbstractSet[str],
             runtime_problem_sizes_dict: Mapping[str, ProblemSizes],
             timing_results_dict: TimingResults):
    """Append measurement results."""
    config = pandas.DataFrame(dict(
      zip(self.config_keys,
          [[expert],
            [self._stringify_types(np_types)],
              [self._stringify_set(dynamic_at_compile_time_sizes)],
              [self._stringify_dict(runtime_problem_sizes_dict)]])))
    results = pandas.DataFrame(dict(
      [(k, timing_results_dict[k]) for k in self.data_keys]))
    product = config.merge(results, how='cross')
    self.data = self.data.append(product)

  def to_dict(self) -> Mapping[str, Any]:
    """Return a dictionary containing the aggregated data."""
    return self.data.to_dict()

  def to_data_frame(self) -> pandas.DataFrame:
    """Return a data frame containing the aggregated data."""
    return self.data

  def plot(self, path: os.path, data_key: str, data_label: str):
    """Plot the provided measurement type for all problem sizes.

    Plot the problem sizes for every expert, np_types, etc. combination.
    """
    config_key_to_plot = "runtime_problem_sizes_dict"
    config_keys_to_fix = [
        k for k in self.config_keys if k is not config_key_to_plot]
    plot_configurations = self.data[config_keys_to_fix].drop_duplicates()
    # Create a plot for every combination of expert, np_types, etc.
    for _, plot_configuration in plot_configurations.iterrows():
      data_to_plot = self._get_data_to_plot(plot_configuration.to_dict())
      # Plot the selected data.
      plt = self._plot_data(config_key_to_plot, data_key,
                            data_label, data_to_plot)
      fig = plt.get_figure()
      fig.tight_layout()
      file_name = self._get_plot_file_name(plot_configuration.to_dict())
      fig.savefig(os.path.join(path, file_name))

  def _plot_data(self,
                 config_key_to_plot: str, data_key: str, data_label: str,
                 data_to_plot: pandas.DataFrame) -> matplotlib.axes.Axes:
    """Plot the provided data and configuration combination."""
    plt = seaborn.violinplot(
        x=config_key_to_plot, y=data_key, data=data_to_plot)
    keys, new_labels = self._compress_problem_sizes_label(
        [text.get_text() for text in plt.xaxis.get_ticklabels()])
    plt.set(xticklabels=new_labels)
    plt.set(xlabel=str.format(
        f"problem sizes [{','.join(keys)}]"), ylabel=data_label)
    plt.tick_params(axis='x', rotation=20)
    return plt

  def _get_data_to_plot(self,
              plot_configuration: Mapping[str, str]) -> pandas.DataFrame:
    """Return the data points for the given plot configuration."""
    data_to_plot = self.data
    for k, v in plot_configuration.items():
      data_to_plot = data_to_plot[data_to_plot[k] == v]
    return data_to_plot

  def _get_plot_file_name(self,
              plot_configuration: Mapping[str, str]) -> str:
    """"Return unique file name for the plot configuration.

    Concat the plot configuration key value pairs and remove special
    characters that may not be supported by the file system.

    Example:
    {'expert': 'SingleTilingExpert', 'np_types': 'float32,float32,float32'}
    ->
    'plot_expert_SingleTilingExpert_np_types_float32float32float32.pdf'
    """
    file_name = "plot"
    for k, v in plot_configuration.items():
      alphanumeric = ''.join([c for c in v if c.isalnum()])
      file_name += str.format(f"_{k}_{alphanumeric}")
    file_name += ".pdf"
    return file_name

  def _compress_problem_sizes_label(self,
              labels: Sequence[str]) -> (Sequence[str], Sequence[str]):
    """Shorten the problem size lables by removing redundant information.

    Plotting the entire problem size configuration for every axis tick
    requires a lot of space and results in overlapping labels. The method
    identifies the dimensions that take different values and filters out
    the dimensions that are constant for the entire plot. Additionally,
    the dimension names (it suffices to plot them once) and sizes values
    are returned seperately.

    Example:
    ["H=64,W=32", "H=64,W=64"]
    ->
    ["W"], ["32", "64"]
    """
    label_dicts = []
    for label in labels:
      groups = re.findall(r"""([a-zA-Z]+)=(\d+|\[[0-9, ]+\])""", label)
      label_dicts.append(dict(groups))
    # Collect all values for a specific key.
    value_dict = {}
    for label_dict in label_dicts:
      for k, v in label_dict.items():
        if k in value_dict:
          value_dict[k].add(v)
        else:
          value_dict[k] = set([v])
    # Collect the keys that have multiple values.
    keys = []
    for k, v in value_dict.items():
      if len(v) != 1:
        keys.append(k)
    # Collect the keys for every label
    new_labels = []
    for label_dict in label_dicts:
      new_labels.append(",".join([label_dict[k] for k in keys]))
    return keys, new_labels

  def _stringify_types(self, value: Sequence[np.dtype]) -> str:
    return ",".join([
      repr(dt).lstrip("<class 'numpy.").rstrip("'>") for dt in value])

  def _stringify_set(self, value: AbstractSet[str]) -> str:
    return ",".join([k for k in value])

  def _stringify_dict(self, value: Mapping[str, ProblemSizes]) -> str:
    return ",".join([
      str.format(f"{k}={v}") for k, v in value.items()])


def _compute_quantiles(measurements: Sequence[float],
                       n_iters: int) -> Sequence[float]:
  return [
    measurements[0],
    measurements[((n_iters * 1) // 100)],
    measurements[((n_iters * 10) // 100)],
    measurements[((n_iters * 25) // 100)],
    measurements[((n_iters * 50) // 100)],
    measurements[((n_iters * 75) // 100)],
    measurements[((n_iters * 90) // 100)],
    measurements[((n_iters * 99) // 100)],
    measurements[-1]
  ]


def timed_invoke(run_n_iters: Callable, gflop_count: float, gbyte_count: float,
                 n_iters: int) -> TimingResults:
  elapsed_ns = run_n_iters(n_iters)
  elapsed_s_per_iter = [sec for sec in np.flip(np.sort(elapsed_ns / 1.e9))]
  gbyte_per_s_per_iter = [(gbyte_count / sec) for sec in elapsed_s_per_iter]
  gflop_per_s_per_iter = [(gflop_count / sec) for sec in elapsed_s_per_iter]
  print(f'xxxxxxxxxx : {n_iters} iters time on {1} threads')
  line = '-' * 120
  header_data = \
      ['slowest', 'p1', 'p10', 'p25', 'p50', 'p75', 'p90', 'p99', 'fastest']
  data = [
      header_data + ['unit'],
      _compute_quantiles(elapsed_s_per_iter, n_iters) + ['seconds'],
      _compute_quantiles(gflop_per_s_per_iter, n_iters) + ['GFlops/s'],
      _compute_quantiles(gbyte_per_s_per_iter, n_iters) + ['GBs/s']
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


def _parse_dimension_list(argument: str) -> Sequence[str]:
  """Parse a sequence of dimensions or the empty list.

  Examples:
  k,m -> ['k', 'm']
  [] -> []
  """
  if argument == '[]':
    return []
  return argument.split(',')


def test_argparser(benchmark_name: str,
                   default_problem_sizes_list: Sequence[Sequence[int]],
                   default_expert_list: Sequence[str],
                   default_dynamic_at_compile_time_list: Sequence[Sequence[str]],
                   default_spec_list: Sequence[str]) -> argparse.Namespace:
  """Test argument parser.

  Creates an argument parser and returns the parsed arguments.

  Arguments:
  benchmark_name: Benchmark name.
  default_problem_sizes_list: Default problem sizes.
  default_expert_list: Default experts.
  default_dynamic_at_compile_time_list: Default dynamic at compile time dimensions.
  default_spec_list: Default specification list.
  """
  parser = argparse.ArgumentParser(description=benchmark_name)
  parser.add_argument('--problem_sizes_list', '-p',
                      type=_parse_problem_sizes,
                      nargs='+',
                      help='problem sizes (e.g., -p 32,32,64 8,8,8)',
                      default=default_problem_sizes_list)
  parser.add_argument('--expert_list', '-e', type=str, nargs='+',
                      help='experts (e.g., -e Expert1 Expert2)',
                      default=default_expert_list)
  parser.add_argument('--dynamic_at_compile_time_list', '-r',
                      type=_parse_dimension_list,
                      nargs='+',
                      help='dynamic at compile time dimensions (e.g., -r k,m k [])',
                      default=default_dynamic_at_compile_time_list)
  parser.add_argument('--spec_list', '-s',
                      type=str,
                      nargs='+',
                      help='problem specifications (e.g., -s mk,kn km,kn)',
                      default=default_spec_list)
  return parser.parse_args(sys.argv[1:])


def test_sizes(dim_names: Sequence[str],
               problem_sizes: ProblemSizes) -> Mapping[str, ProblemSizes]:
  """Annotate the problem size arguments with the given dimension names."""
  return [{k: v for k, v in zip(dim_names, sizes)} for sizes in problem_sizes]


def test_experts(
        all_experts: Sequence[TransformationList],
        all_names: Sequence[str] = [],
        expert_list: Sequence[str] = []) -> Mapping[str, TransformationList]:
  """Annotate the experts with either a provided or a generated name."""
  # Generate the names if none are provided.
  if len(all_experts) != len(all_names):
    all_names = [str(expert) for expert in all_experts]
  # Only filter if the expert list is non empty.
  if not expert_list:
    expert_list = all_names
  return {k: v for k, v in zip(all_names, all_experts) if k in expert_list}


def test_harness(problem_factory: Callable[
    [Mapping[str, Any], Sequence[np.dtype]], ProblemDefinition],
                 np_types_list: Sequence[Sequence[np.dtype]],
                 problem_sizes_list: Sequence[Mapping[str, Any]],
                 experts: Mapping[str, TransformationList],
                 n_iters: int = 1,
                 function_name: str = 'tested_function',
                 dynamic_at_compile_time_sizes: AbstractSet[str] = set(),
                 **kwargs) -> Measurements:
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
  experts: A dictionary of compilation experts to try.
  n_iters: Number of times to run the test.
  function_name: Name of the function in which the IR is emitted, this name can
   be used by compilation experts to target the transformation.
  dynamic_at_compile_time_sizes: A set of size keys that should be treated as unknown (-1)
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
  plot_path: A path to an existing directory to dump the performance plots.

  Returns: A dictionary of all collected benchmark results.
  """

  measurements = Measurements()

  for np_types in np_types_list:
    for problem_sizes_dict in problem_sizes_list:
      compile_time_problem_sizes_dict = {
          key: (value if key not in dynamic_at_compile_time_sizes else -1)
          for key, value in problem_sizes_dict.items()
      }
      runtime_problem_sizes_dict = problem_sizes_dict

      # Init printing.
      print(
          f'\n###############################################################\n'
          f'Compile-time problem size {compile_time_problem_sizes_dict}\n'
          f'Runtime problem size {runtime_problem_sizes_dict}\n'
          f'Problem types {np_types}')
      for name, expert in experts.items():
        print(f'\nCompilation expert {name}')
        problem_definition = problem_factory(problem_sizes_dict, np_types)
        problem = ProblemInstance(problem_definition, np_types)

        problem.compile(
            entry_point_name='main',
            fun_to_benchmark_name=function_name,
            compile_time_problem_sizes_dict=compile_time_problem_sizes_dict,
            transform=expert,
            dump_ir_to_file=kwargs.get('dump_ir_to_file', ''))

        timing_results = problem.run(
            n_iters=n_iters,
            entry_point_name='main',
            runtime_problem_sizes_dict=runtime_problem_sizes_dict,
            dump_obj_to_file=kwargs.get('dump_obj_to_file', ''))

        measurements.append(name, np_types, dynamic_at_compile_time_sizes,
                            runtime_problem_sizes_dict, timing_results)

      problem_definition = problem_factory(problem_sizes_dict, np_types)
      gflops = problem_definition.gflop_count_builder(problem_sizes_dict)
      gbytes = problem_definition.gbyte_count_builder(problem_sizes_dict,
                                                      np_types)

      if 'numpy_benchmark' in kwargs and os.environ.get('BENCHMARK_NUMPY'):
        print('\nNumPy reference\n')
        args = problem_definition.tensors_np_builder(problem_sizes_dict,
                                                     np_types)
        timing_results = timed_invoke(
            lambda n: _run_benchmark_n_iters(kwargs['numpy_benchmark'], n, args,
                                             problem_sizes_dict, np_types),
            gflops, gbytes, n_iters)

        measurements.append('numpy', np_types, dynamic_at_compile_time_sizes,
                            runtime_problem_sizes_dict, timing_results)

      if 'pytorch_benchmark' in kwargs and os.environ.get('BENCHMARK_TORCH'):
        print('\nPyTorch reference\n')
        import torch
        torch.set_num_threads(1)
        numpy_args = problem_definition.tensors_np_builder(
            problem_sizes_dict, np_types)
        args = list(map(torch.from_numpy, numpy_args))
        timing_results = timed_invoke(
            lambda n: _run_benchmark_n_iters(kwargs[
                'pytorch_benchmark'], n, args, problem_sizes_dict, np_types),
            gflops, gbytes, n_iters)

        measurements.append('pytorch', np_types, dynamic_at_compile_time_sizes,
                            runtime_problem_sizes_dict, timing_results)

    if 'plot_path' in kwargs:
      measurements.plot(kwargs.get('plot_path'),
                        'gflop_per_s_per_iter',
                        'compute throughput [GFlop/s]')

    return measurements
