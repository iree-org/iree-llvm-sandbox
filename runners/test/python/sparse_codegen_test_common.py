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
import ctypes
import itertools
import os
from typing import List

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
# TODO: Remove this code when the execution engine is able to do this set up.
_ = ctypes.CDLL(_TEST_LIB, mode=runtime.ctypes.RTLD_GLOBAL)


class InputDesc(object):
  """An input for the operation being tested.

  Attributes:
    _ordering: A list of integers, representing the storage ordering for each
      input dimension.
    _sparsity: A list of DimLevelType, representing the sparsity for each input
      dimension.
    _pointer_bw: The integer bit width for the pointer.
    _index_bw: The integer bit width for the index.
  """

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
  """A test descriptor.

  Attributes:
    _iter_space: Represents the affine expression definition and the size for
      each dimension in the iteration space.
    _inputs: The inputs for the operation being tested. Each input is
      represented by a list of affine expression definitions.
    _output: The output for the operation being tests, represented as a list of
      affine expression definitions.
    _linalg_op: The operation being tested. This is assigned after the object is
      defined because the definition of linalg_op requires other fields in the
      TestDesc object and we can't move the definition of _linalg_op to
      TestDesc.
    _ref_result: The reference result of the test, set up through method
      calculate_reference_result.
  """

  def __init__(self, iter_space_exps: List[dsl.AffineExprDef],
               iter_space_sizes: List[int], output: List[dsl.AffineExprDef],
               *inputs: List[dsl.AffineExprDef]):
    """Constructs a test descriptor.

    Args:
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
      # TODO: We currently use the same data for all the test input. This will
      # be replaced by a dynamic generated input, and the generation method
      # may be specified in the TestDesc object.
      os.environ["TENSOR" + str(index)] = _TEST_DATA_DIR + "mat8.mtx"

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
  def reference_result(self):
    """ Returns the reference result for the test.

    This routine assumes calculate_reference_result has been called to
    calculate the result and record the result in the attribute.
    """
    assert (self._ref_result is not None), \
      "Need to call calculate_reference_result to set up the reference result"
    return self._ref_result

  def get_result(self, p: int, vl: int, input_descs: List[InputDesc]):
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

      # TODO: When vl is non-trivial, enumerates the options for
      # enable-simd-index32.
      si = False
      opt = (f"parallelization-strategy={p} "
             f"vectorization-strategy={v} vl={vl} "
             f"enable-simd-index32={si}")
      compiler = experts.ExpertSparseCompiler(options=opt)

      return self._compile_and_run(compiler, attrs)

  def calculate_reference_result(self):
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

  def _op_boilerplate(self, func_name: str, attrs: List[st.EncodingAttr]):
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

  def _build_module_and_engine(self, compiler: Callable,
                               attrs: List[st.EncodingAttr]):
    """Build the module and the execution engine."""

    module = ir.Module.create()

    # Build the data types for the inputs and output.
    # TODO: Support more data types.
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

  def _compile_and_run(self, compiler: Callable, attrs: List[st.EncodingAttr]):
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


def _sparsities():
  """Enumerates the sparsity values."""
  return [st.DimLevelType.dense, st.DimLevelType.compressed]


def sparsities2():
  """Enumerates the sparsities for an input with rank 2."""
  return itertools.product(_sparsities(), _sparsities())


def orderings2():
  """Enumerates the storage orderings an input with rank 2."""
  return [[0, 1], [1, 0]]


# TODO: add bitwidth 8.
def bitwidths():
  """Enumerates the bit widths to be tested."""
  return [0, 16, 32, 64]


def pars():
  """Enumerates the parallelization option values."""
  return [0, 1, 2, 3, 4]


def vls():
  """Enumerates the vector length option values."""
  return [1, 16, 64]
