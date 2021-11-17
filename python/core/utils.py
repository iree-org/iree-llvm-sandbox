from collections.abc import Callable
from typing import Any, Optional, Sequence, Type

import numpy as np

from mlir.ir import *


################################################################################
# Debug utils.
################################################################################
def inspect(obj):
  print(obj)
  print([name for name in dir(obj) if not name.startswith('__')])


def inspect_all(obj):
  inspect(obj)
  print([name for name in dir(obj) if name.startswith('__')])


def assert_dict_entries_match_keys(dictionary: dict,
                                   required_keys: Sequence[str]):
  assert len(
      set(required_keys).symmetric_difference(set(dictionary.keys()))
  ) == 0, f'dictionary:{dictionary}\n does not contain they exact keys: {required_keys}'


def assert_runtime_sizes_compatible_with_compile_time_sizes(
    runtime_sizes: dict, compile_time_sizes: dict):
  for r, s in zip(runtime_sizes, compile_time_sizes):
    assert s == r or (
        s == -1 and r != -1
    ), f'non-matching compile_time and runtime problem size {s} vs {r}'


################################################################################
# NumPy utils.
################################################################################
def np_type_to_mlir_type(np_type: np.dtype):
  if np_type == np.float16:
    return F16Type.get()
  elif np_type == np.float32:
    return F32Type.get()
  elif np_type == np.float64:
    return F64Type.get()
  else:
    raise Exception(f'unknown scalar type: {np_type}')


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
