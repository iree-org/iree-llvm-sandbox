""" Supports sparse codegen exhaustive tests.

This modules exports two classes InputDesc and TestDesc. It also provides
utilities for enumerating codegen options as well as sparsity annotation for
tensors with certain ranks.

A typical test first constructs an object of TestDesc to describe the operation
being tested. It then calls TestDesc.calculate_reference_result method to set up
the reference result for the test. After this, it enumerates all the
combinations of sparsity annotations and codegen options in a loop, and uses
TestDesc.get_result to run the test with a given set of sparsity annotation and
codegen option.
"""

from collections.abc import Callable
from enum import Enum
from typing import Any, BinaryIO, Iterable, List, Optional, Tuple
import ctypes
import itertools
import os
import random
import tempfile

import numpy as np

# Import MLIR related modules.
from mlir import execution_engine as ee
from mlir import ir
from mlir import runtime
from mlir.dialects import builtin
from mlir.dialects import sparse_tensor as st
from mlir.dialects.linalg.opdsl import lang as dsl

# Import compilation utilties for the tests.
import experts

# We currently rely on an environment to pass in the location for a supporting
# library and assume the test input data are in the same path.
_TEST_LIB = os.getenv("SUPPORTLIB")
_TEST_LIB_DIR = _TEST_LIB[0:_TEST_LIB.rfind("/")]
_TEST_DATA_DIR = _TEST_LIB_DIR + r"/data/"
# Load the supporting library that provides C functions to support the execution
# of the compiler generated code for the operations being tested, such as to
# read and write sparse tensors, and to print vectors.
# TODO(b/195340661): Remove this code when the execution engine is able to do
# this set up.
_ = ctypes.CDLL(_TEST_LIB, mode=runtime.ctypes.RTLD_GLOBAL)


class _Scheme(Enum):
  """Schemes for generating non-zero values for sparse input tensors.

  * `DOT`: A scheme for generating non-zero values as scattered dots.
  * `PLANE`: A scheme for generating non-zero values in hyperplanes.
  """
  DOT = 0
  PLANE = 1


class _Generator(object):
  """Generating values for a sparse input tensor."""

  # A generator has the following attributes:
  # _scheme: A Enum value of _Scheme, representing the generation scheme to be
  #   used.
  # _shape: A list of integers, representing the shape of the input tensor.
  # _values: A list of integers used as the non-zero values for the sparse input
  #   tensor.

  def __init__(self,
               shape: List[int],
               scheme: Optional[_Scheme] = None,
               values: Optional[List[int]] = None):
    """Constructs an input descriptor.

    Args:
      shape: A list of integers, representing the shape of the input tensor.
      scheme: An Enum value of _Scheme, representing the scheme to be used. If a
        scheme is not provided, a scheme is chosen randomly.
      values: A list of integers used cyclically as the non-zero values for the
        sparse input tensor.
    """
    self._shape = shape
    # If a scheme is not specified, randomly choose a scheme.
    random_int = random.randrange(2)
    scheme = scheme or _Scheme.DOT if random_int == 0 else _Scheme.PLANE
    # For tensors with rank <=2, _Scheme.PLANE degenerates to _Scheme.DOT.
    self._scheme = _Scheme.DOT if len(shape) <= 2 else scheme

    self._values = values or [1, 2, 3, 4, 5]

  def generate(self, filename: str) -> None:
    """ Generates the input data and writes it to the given file."""
    # TODO(b/195340661): Use buffers to pass the generated data when the
    # execution engine runtime supports this.
    with open(filename, "w") as file:
      # Output the generation scheme as a comment.
      file.write(f"# scheme={self._scheme}\n")

      # Generate and return the data in a coordinate format.
      data = (
          self._generate_dot(file)
          if self._scheme == _Scheme.DOT else self._generate_plane(file))

      # Output rank and total_elements.
      file.write(f"{len(self._shape)} {len(data)}\n")

      # Output the size for each dimension.
      for d in self._shape:
        file.write(f"{d} ")
      file.write("\n")

      # Output each element using format [coordinates value].
      for e in data:
        for v in e[:-1]:
          # Coordinate format used by the test starts from 1 not 0.
          file.write(f"{v+1} ")
        # Output the value.
        file.write(f" {e[-1]}\n")

  def _generate_dot(self, file: BinaryIO) -> List[int]:
    """Generates a tensor with non-zero values as scattered dots."""
    num_elements = np.prod(self._shape)
    # Generate a non-zero every n values to achieve 20% density.
    n = max(num_elements // 20, 1)
    # Randomdize the position of the first non-zero value.
    first = random.randrange(n)
    # Output these two parameters as a comment.
    file.write(f"# n={n} first={first}\n")

    num_generated = 0
    num_available = len(self._values)
    data = []
    for i in range(num_elements):
      if (i % n) == first:
        ele = self._index_to_coordinate(i, self._shape)
        ele.append(self._values[num_generated % num_available])
        data.append(ele)
        num_generated += 1
    return data

  def _generate_plane(self, file: BinaryIO) -> List[int]:
    """Generates a tensor with non-zero values on a plane."""
    plane_shape = self._shape[-2:]
    other_shape = self._shape[:-2]
    num_plane_elements = np.prod(plane_shape)
    num_other_elements = np.prod(other_shape)
    # Generate a non-zero every n values to achieve 20% density.
    n = max((num_plane_elements * num_other_elements) // 20, 1)
    # Randomdize the position of the first non-zero value on the plane.
    first = random.randrange(n)
    # Output these two parameters as a comment.
    file.write(f"# n={n} first={first}\n")

    num_generated = 0
    num_available = len(self._values)
    data = []
    for j in range(num_other_elements):
      other_coords = self._index_to_coordinate(j, other_shape)
      for i in range(num_plane_elements):
        if (i % n) == first:
          plane_coords = self._index_to_coordinate(i, plane_shape)
          ele = other_coords + plane_coords
          ele.append(self._values[num_generated % num_available])
          data.append(ele)
          num_generated += 1
    return data

  def _index_to_coordinate(self, order: int, shape: List[int]) -> List[int]:
    """Converts a linear index to coordinates."""
    low = 1
    high = order
    coordinates = []
    for dim in shape:
      low *= dim
      coordinates.append(high % low)
      high //= dim
    assert (high == 0 and low == np.prod(shape))
    return coordinates


class InputDesc(object):
  """An input for the operation being tested."""

  # An InputDesc has the following attributes:
  #  _ordering: A list of integers, representing the storage ordering for each
  #    input dimension.
  #  _sparsity: A list of DimLevelType, representing the sparsity for each input
  #    dimension.
  #  _pointer_bw: The integer bit width for the pointer.
  #  _index_bw: The integer bit width for the index.

  def __init__(self, ordering: List[int], sparsity: List[st.DimLevelType],
               pointer_bw: int, index_bw: int):
    """Constructs an input descriptor.

    Args:
      ordering: A list of integers, representing the storage ordering for each
        input dimension.
      sparsity: A list of DimLevelType, representing the sparsity for each input
        dimension.
      pointer_bw: The integer bit width for the pointer.
      index_bw: The integer bit width for the index.

    Raises:
      ValueError: When the lengths of ordering and sparsity differ or when
        ordering is not a permutation of 0..(N-1) where N=len(ordering).
    """

    if len(ordering) != len(sparsity):
      raise ValueError("Different lengths for ordering and sparsity: " +
                       f"{len(ordering)} != {len(sparsity)}.")

    if sorted(ordering) != list(range(len(ordering))):
      raise ValueError("Problem with ordering: " + f"{str(ordering)} != " +
                       f"permutation{str(list(range(len(ordering))))}.")

    self._ordering = ordering
    self._sparsity = sparsity
    self._pointer_bw = pointer_bw
    self._index_bw = index_bw

  @property
  def ordering(self):
    return self._ordering

  @property
  def sparsity(self):
    return self._sparsity

  @property
  def pointer_bw(self):
    return self._pointer_bw

  @property
  def index_bw(self):
    return self._index_bw


class TestDesc(object):
  """A test descriptor."""

  # A TestDesc has the following attributes:
  #  _name: The name of the test.
  #  _iter_space: Represents the affine expression definition and the size for
  #    each dimension in the iteration space.
  #  _inputs: The inputs for the operation being tested. Each input is
  #    represented by a list of affine expression definitions.
  #  _output: The output for the operation being tests, represented as a list of
  #    affine expression definitions.
  #  _linalg_op: The operation being tested. This is assigned after the object
  #    is defined because the definition of linalg_op requires other fields in
  #    the TestDesc object and we can't move the definition of _linalg_op to
  #    TestDesc.
  #  _ref_result: The reference result of the test, set up through method
  #    calculate_reference_result.

  def __init__(self, name: str, iter_space_exps: List[dsl.AffineExprDef],
               iter_space_sizes: List[int], output: List[dsl.AffineExprDef],
               *inputs: List[dsl.AffineExprDef]):
    """Constructs a test descriptor.

    Args:
      name: The name of the test.
      iter_space_exps: A list of AffineExprDef, representing the affine
        expression definition for each dimension in the iteration space.
      iter_space_sizes: A list of integers, representing the size for each
        dimension in the iteration space.
      output: A list of AffineExprDef, representing the affine expression
        definition for each dimension in the output tensor.
      inputs: All the inputs for the operation being tested. Each input is
        represented by a list of AffineExprDef, representing the affine
        expression definition for each dimension in the input tensor.

    Raises:
      ValueError: When there is a problem with the inputs. The lengths of
        iter_space_exps and iter_space_sizes should equal. Affine expressions
        used by output/inputs should be defined in iter_space_exps. Values in
        iter_space_sizes should be larger than zero.
    """
    if len(iter_space_exps) != len(iter_space_sizes):
      raise ValueError("Different lengths for iter_space_exps and " +
                       "iter_space_size: " +
                       f"{len(iter_space_exps)} != {len(iter_space_sizes)}.")

    if any(v <= 0 for v in iter_space_sizes):
      raise ValueError("iter_space_sizes contains values not larger than 0: " +
                       f"{str(iter_space_sizes)}.")

    self._iter_space = dict(zip(iter_space_exps, iter_space_sizes))
    self._linalg_op = None
    self._ref_result = None

    # Verify each affine expression in output.
    for affine in output:
      if affine not in self._iter_space:
        raise ValueError(f"Output affine expression {str(affine)}" +
                         " not defined in the iteration space.")
    self._output = output
    self._name = name

    self._create_data_directory()
    self._inputs = []
    for index, affines in enumerate(inputs):
      # Verify each affine expression in the input.
      for affine in affines:
        if affine not in self._iter_space:
          raise ValueError(f"Input affine expression {str(affine)}" +
                           " not defined in the iteration space.")

      self._inputs.append(affines)
      # The current way of getting a test input is through a filename specified
      # by environment variable TENSOR<N>, where N is the index for the input.
      input_file_path = self._generate_input_data(index)
      os.environ["TENSOR" + str(index)] = input_file_path

  @property
  def inputs(self):
    return self._inputs

  @property
  def output(self):
    return self._output

  @property
  def linalg_op(self):
    return self._linalg_op

  @linalg_op.setter
  def linalg_op(self, op: Callable):
    self._linalg_op = op

  @property
  def get_reference_result(self) -> Iterable[Any]:
    """ Returns the reference result for the test.

    This routine assumes calculate_reference_result has been called to
    calculate the result and record the result in the attribute.
    """
    assert (self._ref_result is not None), \
      "Need to call calculate_reference_result to set up the reference result"
    return self._ref_result

  def get_result(self, p: int, vl: int,
                 input_descs: List[InputDesc]) -> Iterable[Any]:
    """Returns the result for the test with the given codegen parameters.

    Args:
      p: An integer representing the parallelization strategy.
      vl: An integer representing the vector length.
      si: Whether to enable i32 index into vectors.
      input_descs: A list of InputDesc, representing dimension ordering and
        sparsity for the input tensors.

    Returns:
      The result produced by executing the compiled code.
    """
    with ir.Context() as ctx:
      attrs = []
      for desc in input_descs:
        perm = ir.AffineMap.get_permutation(desc.ordering)
        attr = st.EncodingAttr.get(desc.sparsity, perm, desc.pointer_bw,
                                   desc.index_bw)
        attrs.append(attr)

      v = 0 if vl == 1 else 1

      # TODO(b/195340661): When vl is non-trivial, enumerates the options for
      # enable-simd-index32.
      si = False
      opt = (f"parallelization-strategy={p} "
             f"vectorization-strategy={v} vl={vl} "
             f"enable-simd-index32={si}")
      compiler = experts.ExpertSparseCompiler(options=opt)

      return self._compile_and_run(compiler, attrs)

  def calculate_reference_result(self) -> None:
    """Calculates the reference result for the test.

    Returns:
      Uses a default set of codegen parameters to compile the test. Returns the
      result produced by executing the compiled code.
    """
    with ir.Context() as ctx:
      input_descs = []
      for i in range(self._num_inputs()):
        input_descs.append(
            InputDesc(
                self._input_natural_order(i), self._input_sparsity(i), 0, 0))

      self._ref_result = self.get_result(0, 1, input_descs)

  def _input_dim(self, index: int) -> int:
    """Returns the dimension for the given input."""
    return len(self._inputs[index])

  def _input_sparsity(self, index: int) -> List[st.DimLevelType]:
    """Returns the reference sparsity for the given input."""
    return [st.DimLevelType.dense] * self._input_dim(index)

  def _input_natural_order(self, index: int) -> List[int]:
    """Returns the natural ordering for the given input."""
    return list(range(self._input_dim(index)))

  def _num_inputs(self) -> int:
    """Returns the total number of inputs for the operation being tested."""
    return len(self._inputs)

  def _get_dims(self, affine_exps: List[dsl.AffineExprDef]) -> List[int]:
    return [self._iter_space[exp] for exp in affine_exps]

  def _input_dims(self, index: int) -> List[int]:
    """Returns the dimension values for the given input."""
    return self._get_dims(self.inputs[index])

  def _output_dims(self) -> List[int]:
    """Returns the dimension values for the output."""
    return self._get_dims(self.output)

  def _get_type_str(self, affine_exps: List[dsl.AffineExprDef]) -> str:
    dim_strs = [f"{i}x" for i in self._get_dims(affine_exps)]
    return "".join(dim_strs) + "f64"

  def _input_type_str(self, index: int) -> str:
    """Returns the type string representation for the given input."""
    return self._get_type_str(self._inputs[index])

  def _output_type_str(self) -> str:
    """Returns the type string representation for the output."""
    return self._get_type_str(self._output)

  def _create_data_directory(self) -> None:
    """Creates a directory for the generated test inputs."""
    tmp_dir = os.getenv("TEST_TMPDIR", tempfile.gettempdir())
    self._tmp_dir = os.path.join(tmp_dir, self._name)
    if not os.path.exists(self._tmp_dir):
      os.makedirs(self._tmp_dir)

  def _generate_input_data(self, index: int) -> str:
    """Generates data for an input and returns the full path of the file."""
    filename = self._generate_input_filename(index)
    generator = _Generator(self._input_dims(index))
    generator.generate(filename)
    return filename

  def _generate_input_filename(self, index: int) -> str:
    """Generate the full path filename for an input."""
    dim_strs = [str(d) for d in self._input_dims(index)]
    filename = f"tensor{index}_" + "_".join(dim_strs) + ".tns"
    return os.path.join(self._tmp_dir, filename)

  def _op_boilerplate(self, func_name: str,
                      attrs: List[st.EncodingAttr]) -> str:
    """Returns the main method to call the generated sparse kernel."""

    # The function prototype for main.
    output_type_str = self._output_type_str()
    code = f""" func private @getTensorFilename(index) -> !llvm.ptr<i8>
func @main(%c: tensor<{output_type_str}>) -> tensor<{output_type_str}>
  attributes {{ llvm.emit_c_interface }} {{
  %d0 = constant 0.0: f64
"""

    # Set up the input tensors.
    input_setup_strs = []
    for i in range(self._num_inputs()):
      input_type_str = self._input_type_str(i)
      input_setup_strs.append(f"""
  %c{i} = constant {i} : index
  %f{i} = call @getTensorFilename(%c{i}) : (index) -> !llvm.ptr<i8>
  %t{i} = sparse_tensor.new %f{i} : !llvm.ptr<i8> to tensor<{input_type_str},{attrs[i]}>
""")

    code += "".join(input_setup_strs)

    # Call the sparse kernel.
    code += f"%0 = call @{func_name}("
    input_output_strs = []
    for i in range(self._num_inputs()):
      input_output_strs.append(f"%t{i}, ")
    input_output_strs.append("%c) : (")

    for i in range(self._num_inputs()):
      input_type_str = self._input_type_str(i)
      input_output_strs.append(f"tensor<{input_type_str},{attrs[i]}>, ")

    code += "".join(input_output_strs)

    # Return the result.
    code += f"""tensor<{output_type_str}>) -> tensor<{output_type_str}>
  return %0 : tensor<{output_type_str}>
}}"""

    return code

  def _build_module_and_engine(
      self, compiler: Callable,
      attrs: List[st.EncodingAttr]) -> ee.ExecutionEngine:
    """Build the module and the execution engine."""

    module = ir.Module.create()

    # Build the data types for the inputs and output.
    # TODO(b/195340661): Support more data types.
    f64 = ir.F64Type.get()
    inputs_output = []
    for i in range(self._num_inputs()):
      inputs_output.append(
          ir.RankedTensorType.get(self._input_dims(i), f64, attrs[i]))
    inputs_output.append(ir.RankedTensorType.get(self._output_dims(), f64))

    # Build the kernel for the linalg operation being tested.
    with ir.InsertionPoint(module.body):

      @builtin.FuncOp.from_py_func(*inputs_output)
      def linalg_funcop(*args):
        return self._linalg_op(*args[:-1], outs=[args[len(args) - 1]])

    # Invoke JIT compilation.
    compiled_module = compiler(module,
                               self._op_boilerplate("linalg_funcop", attrs))
    engine = ee.ExecutionEngine(compiled_module, 0)

    return engine

  def _compile_and_run(self, compiler: Callable,
                       attrs: List[st.EncodingAttr]) -> Iterable[Any]:
    """Compiles and executes the test."""

    # Feed numpy arrays into MLIR computation.
    output_dims = tuple(self._output_dims())
    c_in = np.zeros(output_dims, np.float64)
    c_in_memref_ptr = runtime.ctypes.pointer(
        runtime.ctypes.pointer(runtime.get_ranked_memref_descriptor(c_in)))
    c_out = np.zeros(output_dims, np.float64)
    c_out_memref_ptr = runtime.ctypes.pointer(
        runtime.ctypes.pointer(runtime.get_ranked_memref_descriptor(c_out)))

    with ir.Context() as ctx, ir.Location.unknown():
      engine = self._build_module_and_engine(compiler, attrs)
      engine.invoke("main", c_out_memref_ptr, c_in_memref_ptr)
      return runtime.ranked_memref_to_numpy(c_out_memref_ptr[0])


# Defines the annotation and codegen options used for the exhaustive tests.


def _sparsities() -> List[st.DimLevelType]:
  """Enumerates the sparsity values."""
  return [st.DimLevelType.dense, st.DimLevelType.compressed]


def sparsities2() -> List[Tuple[st.DimLevelType, st.DimLevelType]]:
  """Enumerates the sparsities for an input with rank 2."""
  return itertools.product(_sparsities(), _sparsities())


def sparsities3(
) -> List[Tuple[st.DimLevelType, st.DimLevelType, st.DimLevelType]]:
  """Enumerates the sparsities for an input with rank 3."""
  return itertools.product(_sparsities(), _sparsities(), _sparsities())


def orderings2() -> List[List[int]]:
  """Enumerates the storage orderings an input with rank 2."""
  return [[0, 1], [1, 0]]


# TODO(b/195340661): Add a method to generate a permutation for range(n) to
# support larger rank values.
def orderings3() -> List[List[int]]:
  """Enumerates the storage orderings for an input with rank 3."""
  return [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]


# TODO(b/195340661): Add bitwidth 8.
def bitwidths() -> List[int]:
  """Enumerates the bit widths to be tested."""
  return [0, 16, 32, 64]


def pars() -> List[int]:
  """Enumerates the parallelization option values."""
  return [0, 1, 2, 3, 4]


def vls() -> List[int]:
  """Enumerates the vector length option values."""
  return [1, 16, 64]


def command_line_parser():
  """Parses the command line arguments and returns the argument parser."""
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-num_processes",
      type=int,
      required=False,
      default=os.cpu_count(),
      help="the number of processes used to run the test (default to os.cpu_count())"
  )
  args = parser.parse_args()
  return args
