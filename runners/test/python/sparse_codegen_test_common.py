""" Supports sparse codegen exhaustive tests.

This modules exports two classes InputDesc and TestDesc. It provides utilities
for enumerating codegen options as well as sparsity annotation for tensors with
certain ranks. It also provides utilities for parsing comman lines, generating
sparse tensor inputs, running tests sequentially or in parallel, and
postprocessing a test run.

A typical test first constructs an object of TestDesc to describe the operation
being tested. It then calls TestDesc.calculate_reference_result method to set up
the reference result for the test. After this, it calls
TestDesc.run_tests_sequential_or_parallel by providing a function to enumerate
all the combinations of sparsity annotations and codegen options and a function
to run the test with one parameter combination. The function that runs the test
for one parameter combination invokes TestDesc.get_result to run the test with
a given set of sparsity annotation and codegen option and compares the result
with the reference result.
"""

from enum import Enum
import dataclasses
from typing import Any, Callable, List, Optional, Tuple
import argparse
import ctypes
import itertools
import logging
import numpy as np
import os
import random

# Import MLIR related modules.
from mlir import execution_engine as ee
from mlir import ir
from mlir import runtime
from mlir.dialects import builtin
from mlir.dialects import sparse_tensor as st
from mlir.dialects.linalg.opdsl import lang as dsl

# Import compilation utilties for the tests.
import experts


# Message to print out when tests stop with failure.
FAILURE_MESSAGE = "FAILURE"

# Generate a non-zero every 5 values to achieve 20% density.
_STEP_FOR_NON_ZERO_VALUES = 5
# The non-zero values used by the generator, if not provided by the tests.
_DEFAULT_NON_ZERO_VALUES = (1, 2, 3, 4, 5)
# A plane has two dimensions.
_RANK_FOR_PLANE = 2
# A default seed to initialize random state.
_DEFAULT_SEED = 5

# The name for the environment variable that provides the full path for the
# supporting library.
_SUPPORTLIB_ENV_VAR = "SUPPORTLIB"
# The default supporting library if the environment variable is not provided.
_DEFAULT_SUPPORTLIB = "libmlir_c_runner_utils.so"
# The JIT compiler optimization level.
_OPT_LEVEL = 2
# The entry point to the JIT compiled program.
_ENTRY_NAME = "main"

# TODO(b/195340661): Add bitwidth 8.
# Bitwidths for pointer and indices.
_SUPPORTED_BIT_WIDTHS = (0, 16, 32, 64)
# Sparse codegen parallelization options.
_SUPPORTED_PARALLELIZATION_OPTIONS = (0, 1, 2, 3, 4)
# Sparse codegen vector lengths.
_SUPPORTED_VECTOR_LENGTHS = (1, 16, 64)
# Available sparsity values for each tensor dimension.
_SUPPORTED_SPARSITY_VALUES = (st.DimLevelType.dense, st.DimLevelType.compressed)


# Alias for annotating the type for the function object used to invoke the
# compiler.
CompilerType = Callable[
    [ir.Module, Callable[[str, List[st.EncodingAttr]], str]], ir.Module]


class _Scheme(Enum):
  """Schemes for generating non-zero values for sparse input tensors.

  * `DOT`: A scheme for generating non-zero values as scattered dots.
  * `PLANE`: A scheme for generating non-zero values in hyperplanes.
  """
  DOT = 0
  PLANE = 1


class TDType(Enum):
  """ The data types being tested."""
  # TODO(b/195340661): Add int8.
  I16 = np.int16
  I32 = np.int32
  I64 = np.int64
  # numpy _ctype_from_dtype_scalar can't handle float16 yet.
  F32 = np.float32
  F64 = np.float64


# Supported integer types.
_SUPPORTED_INT_TYPES = (TDType.I16, TDType.I32, TDType.I64)
# Supported floating point types.
_SUPPORTED_FP_TYPES = (TDType.F32, TDType.F64)
# The prefix for TDType enum string name produced by str(an enum).
_TDTYPE_NAME_PREFIX = TDType.__name__ + "."


def _generate_tensor_dot(shape: List[int], values: Tuple[int, ...],
                         first_nonzero_pos: int) -> List[int]:
  """Generates a tensor with non-zero values as scattered dots."""
  num_elements = np.prod(shape)

  num_generated = 0
  num_available = len(values)
  data = []
  for i in range(num_elements):
    if (i % _STEP_FOR_NON_ZERO_VALUES) == first_nonzero_pos:
      data.append(values[num_generated % num_available])
      num_generated += 1
    else:
      data.append(0)
  return data


def _generate_tensor_plane(shape: List[int], values: Tuple[int, ...],
                           first_nonzero_pos: int) -> List[int]:
  """Generates a tensor with non-zero values on planes."""
  plane_shape = shape[-_RANK_FOR_PLANE:]
  other_shape = shape[:-_RANK_FOR_PLANE]
  num_plane_elements = np.prod(plane_shape)
  num_other_elements = np.prod(other_shape)

  num_generated = 0
  num_available = len(values)
  data = []
  for j in range(num_other_elements):
    for i in range(num_plane_elements):
      if (i % _STEP_FOR_NON_ZERO_VALUES) == first_nonzero_pos:
        data.append(values[num_generated % num_available])
        num_generated += 1
      else:
        data.append(0)
  return data


def generate_tensor(shape: List[int],
                    scheme: Optional[_Scheme] = None,
                    values: Optional[Tuple[int, ...]] = None,
                    seed: int = _DEFAULT_SEED) -> List[int]:
  """Generates values for a sparse input tensor.

  Args:
    shape: A list of integers, representing the dimensions of the input tensor.
    scheme: An Enum value of _Scheme, representing the scheme to be used. If a
      scheme is not provided, a scheme is chosen randomly.
    values: A tuple of integers used cyclically as the non-zero values for
      generating the sparse tensor.
    seed: An integer value to initialize the random number generator state. The
      random number generator is used to select a generation scheme when a
      scheme is not provided and to decide on the position of the first non-zero
      value.

   Returns:
     The sparse tensor value represented as a list of integers.
  """
  random_state = np.random.RandomState(_DEFAULT_SEED)
  if len(shape) <= 2:
    # When rank <= 2, _Scheme.PLANE degenerates to _Scheme.DOT.
    scheme = _Scheme.DOT
  elif scheme is None:
    # If a scheme is not specified, randomly choose a scheme.
    scheme = _Scheme.PLANE if random_state.choice(2) else _Scheme.DOT

  values = values or _DEFAULT_NON_ZERO_VALUES

  # Generate a random value in range 0.._STEP_FOR_NON_ZERO_VALUES to randomdize
  # the position of the first non-zero value.
  first_nonzero_pos = random_state.choice(_STEP_FOR_NON_ZERO_VALUES)

  # Generate the data as a list of values.
  data = (
      _generate_tensor_dot(shape, values, first_nonzero_pos)
      if scheme == _Scheme.DOT else _generate_tensor_plane(
          shape, values, first_nonzero_pos))

  return data


@dataclasses.dataclass(frozen=True)
class InputDesc:
  """Describing an input for the operation being tested.

  Attributes:
    ordering: A list of integers for the storage ordering of the input
      dimensions.
    sparsity: A list of DimLevelType for the sparsity of each input dimension.
    pointed_bw: An integer pointer bit width.
    index_bw: An integer index bit width.
  """
  ordering: List[int]
  sparsity: List[st.DimLevelType]
  pointer_bw: int
  index_bw: int

  def __post_init__(self):
    if len(self.ordering) != len(self.sparsity):
      raise ValueError("Different lengths for ordering and sparsity: " +
                       f"{len(self.ordering)} != {len(self.sparsity)}.")

    if sorted(self.ordering) != list(range(len(self.ordering))):
      raise ValueError("Problem with ordering: " + f"{str(self.ordering)} != " +
                       f"permutation{str(list(range(len(self.ordering))))}.")


def _ctype_pointer_from_array(array) -> ctypes.POINTER:
  """Returns the ctype pointer for the given numpy array."""
  return ctypes.pointer(
      ctypes.pointer(runtime.get_ranked_memref_descriptor(array)))


class TestDesc:
  """Describing a test for an opeartion.

  A test descriptor has the following properties:
    inputs: A read-only property to access the input affine expressions.
    outputs: A read-only property to access the output affine expressions.
    linalg_op: A writable property to access the linear algebra operation
      being test.
  """

  # A TestDesc has the following attributes:
  #  _name: The name of the test.
  #  _iter_space: Represents the affine expression definition and the size for
  #    each dimension in the iteration space.
  #  _input_affines: The list of inputs. Each input for the operation being
  #  tested is defined by a list of affine expression definition.
  #  _input_tensors: The list of input tensors. Each input tensor is represented
  #    as a list of integers.
  #  _output: The output for the operation being tests, represented as a list of
  #    affine expression definitions.
  #  _linalg_op: The operation being tested. This is assigned after the object
  #    is defined because the definition of linalg_op requires other fields in
  #    the TestDesc object and we can't move the definition of _linalg_op to
  #    TestDesc.
  #  _ref_result: The reference result of the test, set up through method
  #    calculate_reference_result.

  @property
  def inputs(self) -> List[List[dsl.AffineExprDef]]:
    """The input affine expressions."""
    return self._input_affines

  def _get_dims_from_affine_expr(
      self, affine_exps: List[dsl.AffineExprDef]) -> List[int]:
    """Returns the dimensions for the affine expression."""
    return [self._iter_space[exp] for exp in affine_exps]

  def _get_input_dims(self, index: int) -> List[int]:
    """Returns the dimension values for the given input."""
    return self._get_dims_from_affine_expr(self.inputs[index])

  def __init__(self, name: str, iter_space_exps: List[dsl.AffineExprDef],
               iter_space_sizes: List[int], output: List[dsl.AffineExprDef],
               *inputs: List[List[dsl.AffineExprDef]]):
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

    self._input_affines = []
    self._input_tensors = []
    for index, affines in enumerate(inputs):
      # Verify each affine expression in the input.
      for affine in affines:
        if affine not in self._iter_space:
          raise ValueError(f"Input affine expression {str(affine)}" +
                           " not defined in the iteration space.")

      self._input_affines.append(affines)
      self._input_tensors.append(generate_tensor(self._get_input_dims(index)))

  @property
  def output(self) -> List[dsl.AffineExprDef]:
    """The output affine expressions."""
    return self._output

  @property
  def linalg_op(self) -> dsl.LinalgOpDef:
    """The linear algebra operation being tested."""
    return self._linalg_op

  @linalg_op.setter
  def linalg_op(self, op: Callable[..., dsl.DefinedOpCallable]) -> None:
    self._linalg_op = op

  def _get_num_inputs(self) -> int:
    """Returns the total number of inputs for the operation being tested."""
    return len(self._input_affines)

  def _get_output_dims(self) -> List[int]:
    """Returns the dimension values for the output."""
    return self._get_dims_from_affine_expr(self.output)

  def _get_inputs_for_type(self, type: TDType) -> List[np.ndarray]:
    """Returns a list of numpy array with the given type for the inputs."""
    return [
        np.array(v,
                 dtype=type.value).reshape(self._get_dims_from_affine_expr(a))
        for v, a in zip(self._input_tensors, self._input_affines)
    ]

  def _get_type_str(self, dims: List[int], type: TDType) -> str:
    """Returns the type string for the given shape and type."""
    dim_strs = [f"{i}x" for i in dims]
    # For a TDType enum, such as I32, its string name is TDType.I32 while its MLIR
    # string name is i32.
    return "".join(dim_strs) + (str(type))[len(_TDTYPE_NAME_PREFIX):].lower()

  def _generate_mlir_program(self, func_name: str, type: TDType,
                             attrs: List[st.EncodingAttr]) -> str:
    """Returns the MLIR text program for the main method to call the function.

    The MLIR text program has this format:
    main (%d0: tensor<xtype>, %d1: tensor<xtype>, %c: tensor<xtype>)
      -> tensor<xtype> attributes { llvm.emit_c_interface } {
      // Set up input tensor
      // Call the function.
      // Return the result of the function call.
    }

    Args:
      func_name: The name of the function for the operation being tested.
      type: The data type used to run the operation being tested.
      attrs: A list of EncodingAttr, one for each input of the operation being
        tested.

    Returns:
      The MLIR program in text format.
    """

    # Construct the MLIR text for the main function header. This includes the
    # input argument names and data types, the output data types, and the
    # attribute for the main function.

    num_inputs = self._get_num_inputs()
    input_type_strs = [
        self._get_type_str(self._get_input_dims(i), type)
        for i in range(num_inputs)
    ]
    # An argument string describes an input name and data type or the output
    # data type. The separator between argument strings will be added when
    # joining the strings.
    argument_strs = []
    for i in range(num_inputs):
      argument_strs.append(f"%d{i}: tensor<{input_type_strs[i]}>")

    output_type_str = self._get_type_str(self._get_output_dims(), type)
    argument_strs.append(f"%c: tensor<{output_type_str}>")
    argument_string = ", ".join(argument_strs)
    code_header = f"func @{_ENTRY_NAME}({argument_string}) " + f"""-> tensor<{output_type_str}>
  attributes {{ llvm.emit_c_interface }} {{"""

    # Construct the MLIR text to set up the inputs for calling the given
    # function. Each input to the function is computed by applying a sparse
    # tensor conversion on an input of the main function.
    input_setup_strs = []
    for i in range(num_inputs):
      input_setup_strs.append(
          f"  %t{i} = sparse_tensor.convert %d{i} : tensor<{input_type_strs[i]}> to tensor<{input_type_strs[i]},{attrs[i]}>"
      )

    # Start each input setup in a new line.
    code_input_setup = "\n".join(input_setup_strs)

    # Construct the MLIR text to call the given function.

    # The separator between input name strings will be added when joining the
    # strings.
    input_name_strs = []
    for i in range(num_inputs):
      input_name_strs.append(f"%t{i}")
    input_name_strs.append("%c) : (")
    input_name_string = ", ".join(input_name_strs)

    # The separator between input/output type strings will be added when joining
    # the strings.
    input_output_type_strs = []
    for i in range(num_inputs):
      input_output_type_strs.append(f"tensor<{input_type_strs[i]},{attrs[i]}>")
    input_output_type_strs.append(
        f"tensor<{output_type_str}>) -> tensor<{output_type_str}>")
    input_output_type_string = ", ".join(input_output_type_strs)

    code_call = f"  %0 = call @{func_name}(" + input_name_string + input_output_type_string

    # Construct the MLIR text to return the result and mark the ending of the
    # program.
    code_return = f"""  return %0 : tensor<{output_type_str}>
}}"""

    # Use a blank line to separate different parts of the MLIR text program.
    return "\n".join([code_header, code_input_setup, code_call, code_return])

  def _build_module_and_engine(
      self, compiler: CompilerType, type: TDType,
      attrs: List[st.EncodingAttr]) -> ee.ExecutionEngine:
    """Builds the program and the execution engine.

    Args:
      compiler: A Callable object for invoking the compiler.
      type: The data type for the operation being tested.
      attrs: A list of EncodingAttr, one for each input of the operation being
        tested.

    Returns:
      The execution engine that executes the JIT compiled operation.
    """

    module = ir.Module.create()

    # Build the data types for the inputs and output.
    tdtype_to_irtype = {
        TDType.I16: ir.IntegerType.get_signless(16),
        TDType.I32: ir.IntegerType.get_signless(32),
        TDType.I64: ir.IntegerType.get_signless(64),
        TDType.F32: ir.F32Type.get(),
        TDType.F64: ir.F64Type.get()
    }
    ir_type = tdtype_to_irtype[type]
    inputs_output = []
    for i in range(self._get_num_inputs()):
      inputs_output.append(
          ir.RankedTensorType.get(self._get_input_dims(i), ir_type, attrs[i]))
    inputs_output.append(
        ir.RankedTensorType.get(self._get_output_dims(), ir_type))

    # Build the kernel for the linalg operation being tested.
    with ir.InsertionPoint(module.body):

      @builtin.FuncOp.from_py_func(*inputs_output)
      def linalg_funcop(*args):
        return self._linalg_op(*args[:-1], outs=[args[len(args) - 1]])

    # Invoke JIT compilation.
    compiled_module = compiler(
        module, self._generate_mlir_program(linalg_funcop.__name__, type,
                                            attrs))

    # We currently rely on an environment to pass in the full path for a
    # supporting library to overwrite the default supporting library.
    support_lib = os.getenv(_SUPPORTLIB_ENV_VAR, _DEFAULT_SUPPORTLIB)
    engine = ee.ExecutionEngine(
        compiled_module, opt_level=_OPT_LEVEL, shared_libs=[support_lib])

    return engine

  def _compile_and_run(self, compiler: CompilerType, type: TDType,
                       attrs: List[st.EncodingAttr],
                       inputs: List[np.ndarray]) -> np.ndarray:
    """Compiles and executes the test.

    Args:
      compiler: A Callable object for invoking the compiler.
      attrs: A list of EncodingAttr, one for each input of the operation being
        tested.
      inputs: A list of numpy arrays for the input tensors.

    Returns:
      The output of the operation being test, represented as a numpy array.
    """
    # Numpy arrays are accessed by MLIR computation via their ctype pointers.
    # Gather a list of ctype pointers for the numpy arrays.
    ctype_pointers = []
    output_dims = self._get_output_dims()

    # Add the pointer for the output tensor.
    c_out = np.zeros(output_dims, type.value)
    ctype_pointers.append(_ctype_pointer_from_array(c_out))

    # Add the pointers for the input tensors.
    for i in range(self._get_num_inputs()):
      ctype_pointers.append(_ctype_pointer_from_array(inputs[i]))

    # Add the pointer for the initial value of the output tensor. Currently,
    # the initial value and the output value have to be different.
    c_init = np.zeros(output_dims, type.value)
    ctype_pointers.append(_ctype_pointer_from_array(c_init))

    # Invoke JIT compilation, then execute the compiled code.
    with ir.Context() as ctx, ir.Location.unknown():
      engine = self._build_module_and_engine(compiler, type, attrs)
      engine.invoke(_ENTRY_NAME, *ctype_pointers)
      return runtime.ranked_memref_to_numpy(ctype_pointers[0][0])

  def get_result(self, p: int, vl: int, type: TDType,
                 input_descs: List[InputDesc]) -> np.ndarray:
    """Returns the result for the test for the given codegen parameters.

    Args:
      p: An integer representing the parallelization strategy.
      vl: An integer representing the vector length.
      type: The TDType for the result.
      input_descs: A list of InputDesc, representing dimension ordering and
        sparsity for the input tensors.

    Returns:
      The result produced by executing the compiled code.
    """
    with ir.Context() as ctx:
      inputs = self._get_inputs_for_type(type)
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

      return self._compile_and_run(compiler, type, attrs, inputs)

  def calculate_reference_result(self, type: TDType) -> None:
    """Calculates the reference result for the test.

    Args:
      type: The data type used to run the operation to get the reference result.

    Returns:
      Uses a default set of codegen parameters to compile the test. Returns the
      result produced by executing the compiled code.
    """
    with ir.Context() as ctx:
      input_descs = []
      for i in range(self._get_num_inputs()):
        input_descs.append(
            InputDesc(
                list(range(len(self._input_affines[i]))),
                [st.DimLevelType.dense] * len(self._input_affines[i]), 0, 0))

      self._ref_result = self.get_result(0, 1, type, input_descs)

  def get_reference_result(self, type: TDType) -> np.ndarray:
    """ Returns the reference result for the test.

    This routine assumes calculate_reference_result has been called to
    calculate the result and record the result in the attribute.

    Args:
      type: The data type for the output result.

    Returns:
      Converts the pre-calculated reference result to the desired data type and
      returns the result.

    Raises:
      ValueError: if calculate_reference_result is not called to make the
        reference result available.
    """

    if self._ref_result is None:
      raise ValueError("Need to call calculate_reference_result to set up" +
                       " the reference result.")

    return self._ref_result.astype(type.value)


# Defines the annotation and codegen options used for the exhaustive tests.


def sparsities2() -> List[Tuple[st.DimLevelType, st.DimLevelType]]:
  """Enumerates the sparsities for an input with rank 2."""
  return itertools.product(_SUPPORTED_SPARSITY_VALUES,
                           _SUPPORTED_SPARSITY_VALUES)


def sparsities3(
) -> List[Tuple[st.DimLevelType, st.DimLevelType, st.DimLevelType]]:
  """Enumerates the sparsities for an input with rank 3."""
  return itertools.product(_SUPPORTED_SPARSITY_VALUES,
                           _SUPPORTED_SPARSITY_VALUES,
                           _SUPPORTED_SPARSITY_VALUES)


# TODO(b/195340661): Add a method to generate a permutation for range(n) to
# support larger rank values. This will retire the use of the constant values.
def orderings2() -> List[List[int]]:
  """Enumerates the storage orderings an input with rank 2."""
  return [[0, 1], [1, 0]]


# TODO(b/195340661): Add a method to generate a permutation for range(n) to
# support larger rank values.  This will retire the use of the constant values.
def orderings3() -> List[List[int]]:
  """Enumerates the storage orderings for an input with rank 3."""
  return [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]


# TODO(b/195340661): Add bitwidth 8.
def bitwidths() -> Tuple[int, ...]:
  """Enumerates the bit widths to be tested."""
  return _SUPPORTED_BIT_WIDTHS


def pars() -> Tuple[int, ...]:
  """Enumerates the parallelization option values."""
  return _SUPPORTED_PARALLELIZATION_OPTIONS


def vls() -> Tuple[int, ...]:
  """Enumerates the vector length option values."""
  return _SUPPORTED_VECTOR_LENGTHS


def int_types() -> Tuple[TDType, ...]:
  """Enumerates the integer data types to be tested."""
  return _SUPPORTED_INT_TYPES


def fp_types() -> Tuple[TDType, ...]:
  """Enumerates the floating point data types to be tested."""
  return _SUPPORTED_FP_TYPES


def all_types() -> Tuple[TDType, ...]:
  """Enumerates all the data types to be tested."""
  return _SUPPORTED_INT_TYPES + _SUPPORTED_FP_TYPES


def supported_tensor_types(type: TDType, pw: int, iw: int):
  """ Checks whether the tensor type combination is supported.

  Args:
    type: A TDType enum for the data type of the tensor values.
    pw:   The pointer bitwidth for the tensor storage representation.
    iw:   The index bitwidth for the tensor storage representation.

  Returns:
    A boolean value to indicate whether the combination is supported (True) or
    not supported (False).
  """
  # newSparseTensor only supports pw == iw for integer types. For int64, it only
  # supports pw == iw == 64.
  return (type
          not in _SUPPORTED_INT_TYPES) or (pw == iw and
                                           (type != TDType.I64 or pw == 64))


def _get_command_line_values() -> Tuple[int, int]:
  """Parses the command line and returns (num_processes, log_level)."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-num_processes",
      type=int,
      required=False,
      default=os.cpu_count(),
      help="the number of processes to run the test (default os.cpu_count())")
  parser.add_argument(
      "-log",
      choices=["info", "error"],
      default="info",
      help="the logging level (default=info)"),
  args = parser.parse_args()
  levels = {
      "error": logging.ERROR,
      "info": logging.INFO,
  }
  return args.num_processes, levels[args.log]


def _run_tests_sequential(parameter_combinations: Callable[[], Tuple[Any, ...]],
                          run_test: Callable[..., bool]) -> bool:
  """Tests all combinations sequentially."""
  return all(run_test(*c) for c in parameter_combinations())


def _run_tests_parallel(num_processes: int,
                        parameter_combinations: Callable[[], Tuple[Any, ...]],
                        run_test: Callable[..., bool]) -> bool:
  """Tests all combinations in parallel with the given number of processes."""

  # Python multiprocessing doesn't work within Google. Import the Pool module
  # only when multiprocessing is enable for this reason.
  from multiprocessing import Pool
  with Pool(num_processes) as pool:
    # For each combination, assign a job to the worker pool and return a
    # placeholder object for getting the test result. We use `c` not `*c` here
    # as apply_async unpacks the tuple.
    result_objs = [
        pool.apply_async(run_test, c) for c in parameter_combinations()
    ]

    # Get the results of the tests using the placeholder objects.
    return all(result.get() for result in result_objs)


def run_tests_sequential_or_parallel(num_processes: int,
                                     parameter_combinations: Callable[[], Tuple[
                                         Any, ...]],
                                     run_test: Callable[..., bool]) -> bool:
  """Runs the tests with the given number of processes.

  Args:
    num_processes: An integer for the number of processes used to run the tests.
      The tests are run in parallel when this value is larger than one.
    parameter_combinations: A Callable object for generating all the
      combinations of the parameter values used to invoke run_test.
    run_test: A Callable object for running a test with a given combination of
      parameter values.

  Returns:
    A boolean to indicate whether all tests pass (True) or there are failing
      tests (False).
  """
  return (_run_tests_sequential(parameter_combinations, run_test)
          if num_processes <= 1 else _run_tests_parallel(
              num_processes, parameter_combinations, run_test))


def get_num_processes_and_run_tests(module_name: str,
                                    test_driver: Callable[[int], bool]) -> bool:
  """Determines the number of processes and invokes the test driver.

  The tests run differently in OSS vs in Google for two reasons.
  - In Google, we use a python script to load and execute the module that
    contains the tests to support the loading of the MLIR libraries. In OSS, we
    directly run the module that contains the tests.
  - Python multiprocessing works in OSS but doesn't work in Google.
  As such, we only enable the commandline parser and multiprocessing when the
  module is run directly.

  Args:
    module_name: The __name__ of the module that contains the tests, used to
      determine whether the module is run directly or not.
    test_driver: A callable object to run all tests in the module with the given
      number of processes.

  Returns:
    A boolean to indicate whether all tests pass (True) or there are failing
      tests (False).
  """
  if module_name != "__main__":
    num_processes = 1
    log_level = logging.INFO
  else:
    num_processes, log_level = _get_command_line_values()

  logging.basicConfig(level=log_level)
  return test_driver(num_processes)


def test_combination_wrapper(
    test_combination: Callable[..., bool]) -> Callable[..., int]:
  """Wraps a test function with post processing functionality.

  In particular, the wrapper invokes test_combination, logs the test and its
  status, and returns a boolean to indicate the status of passing (True) or
  failing (False).

  Args:
    test_combination: A Callable object for invoking the test with one
      combination of the test parameters, and returns a boolean to indicate the
      status of passing (True) or failing (False).

  Returns:
    A wrapper of the given test_combination function.
  """

  def wrapper(*args) -> int:
    passed = test_combination(*args)

    status_str = "passed" if passed else "failed"
    test_name = "_".join([str(i) for i in args])
    logging.info(f"test_{test_name} {status_str}.")

    return passed

  return wrapper
