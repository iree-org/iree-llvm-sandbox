import time
import argparse
import json
import numpy as np
from itertools import chain

from mlir.dialects import linalg
from mlir.dialects.linalg.opdsl.lang import OperandKind
from mlir.runtime import *

from ..core.compilation import numpy_type, compile_and_callback
from ..core.search_vars import collect_variables
from ..core import experts


def parse_args(argv):
  parser = argparse.ArgumentParser(description='Command-line directed search.')
  parser.add_argument(
      '--op',
      type=str,
      required=True,
      help='Name of the linalg op to instantiate.')
  parser.add_argument(
      '--expert',
      type=str,
      required=True,
      help='Name of the expert to use for compilation.')
  parser.add_argument(
      '--assign',
      type=str,
      help='A json dictionary of key-value pairs to specify op or expert variables.'
  )
  parser.add_argument(
      '--iters',
      type=int,
      default=100,
      help='Number of iterations of the MLIR loop.')
  parser.add_argument(
      '--runs',
      type=int,
      default=10,
      help='Number of times the MLIR program is run to measure runtime.')
  return parser.parse_args(argv[1:])


def validate_args(args):
  no_errors = True

  def error(msg):
    nonlocal no_errors
    no_errors = False
    print(msg)

  if not hasattr(linalg, args.op):
    error(f'Unknown op: {args.op}.')

  if not hasattr(experts, args.expert):
    error(f'Unknown expert name: {args.expert}')

  op = getattr(linalg, args.op)
  expert = getattr(experts, args.expert)
  variables = collect_variables(op)
  variables.update(expert.variables)
  assignments = json.loads(args.assign)
  for var_name in assignments.keys():
    if var_name not in assignments:
      error(f'Variable {variable.name} was not assigned.')

  iters = args.iters
  if iters < 0:
    error(f'Number of iterations must be non-negative.')

  runs = args.runs
  if runs < 0:
    error(f'Number of runs must be non-negative.')

  if no_errors:
    return (op, expert, assignments, iters, runs)
  else:
    return None


def invoke(op, expert, assignments, iters, runs):

  def section(name):
    print(f'--- {name}')

  def timed(callback, *args):
    start = time.time()
    callback(*args)
    end = time.time()
    return end - start

  def random_array_inputs():
    results = []
    for odef in op.model.registered_operands.values():
      assert (odef.kind == OperandKind.InputTensor or
              odef.kind == OperandKind.OutputTensor)
      np_type = numpy_type(assignments[odef.type_var.name])
      shape = [assignments[sym.symname] for sym in odef.size_exprs]
      arr0 = np.random.rand(*shape)
      arr = arr0.astype(np_type)
      results.append(arr)
    return results

  def to_memref_ptr(arr):
    memref_descr = get_ranked_memref_descriptor(arr)
    return ctypes.pointer(ctypes.pointer(memref_descr))

  def measure_runtime(execution_engine):
    array_inputs = random_array_inputs()
    memref_inputs = list(map(to_memref_ptr, array_inputs))
    index_ptr_t = ctypes.c_longlong * 1

    def invoke(iters):
      execution_engine.invoke('main', *memref_inputs, index_ptr_t(iters))

    # Dry-run.
    timed(invoke, 1)

    # Measure.
    times = []
    for _ in range(runs):
      times.append(timed(invoke, iters))

    # Report best of the runs.
    return min(times)

  def callback(module, execution_engine):
    section('mlir')
    print(module)

    if iters > 0 and runs > 0:
      elapsed_time = measure_runtime(execution_engine)
      section('runtime')
      print(f'time: {elapsed_time}')
      print(f'iters: {iters}')
      print(f'throughput: {iters/elapsed_time}')

  compile_and_callback(
      op,
      expert(
          'matmul_on_tensors',
          op.op_name,
          print_ir_after_all=True,
          **assignments), callback, **assignments)


def main(argv):
  args = parse_args(argv)
  validated = validate_args(args)
  if validated is None:
    return
  invoke(*validated)


if __name__ == '__main__':
  import sys
  main(sys.argv)
