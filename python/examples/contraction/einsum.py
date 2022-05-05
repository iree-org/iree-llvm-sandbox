from iree.compiler.ir import *
from iree.compiler.dialects.linalg.opdsl.lang import *

import itertools


class EinsumSpecification:
  """Structured representation of a string einsum specification."""

  def __init__(self, specification: str, domain: str):
    """Creates a specification given its string format."""
    # Split out the operands and the result part of the specification.
    specification_split = specification.split("->")
    assert len(specification_split
              ) <= 2, "Expected at most one '->' in the specification."
    operands = specification_split[0]
    result = specification_split[1] if len(specification_split) > 1 else None

    # Split the operands.
    operands_split = operands.split(",")
    assert len(
        operands_split
    ) <= 2, "Expected at most two comma-separated operands in the specification."

    # Verify the operands contain no duplicates and are all lower case.
    operand_dims = [None, None]
    for idx, dims in enumerate(operands_split):
      dims.strip()
      assert len(set(dims)) == len(dims), str.format(
        f"Unexpected duplicate symbol in operand {idx}.")
      assert dims.islower(), str.format(
        f"Expected only lowercase symbols in operand {idx}.")
      operand_dims[idx] = dims
    lhs_dims, rhs_dims = operand_dims[0], operand_dims[1]

    # Infer the output dimensions for two-operand specifications. The output
    # dimensions are the symbols unique to lhs and rhs, in alphabetical order.
    inferred_output_dims = None
    if rhs_dims is not None:
      reduction_dims = set(lhs_dims).intersection(rhs_dims)
      all_dims = set(lhs_dims).union(rhs_dims)
      inferred_output_dims = sorted(all_dims.difference(reduction_dims))

    # Use the inferred output specification if there is no user-specified one.
    if result is not None:
      result = result.strip()
      output_dims = result
      if len(set(output_dims)) != len(output_dims):
        raise NotImplementedError("Repeated output dimensions.")
      if inferred_output_dims is not None:
        assert set(result) == set(inferred_output_dims), str.format(
          f"Expected output to match inferred output {inferred_output_dims}.")
    else:
      assert inferred_output_dims is not None, "Expected output specification."
      output_dims = "".join(inferred_output_dims)

    # Use the specified iteration domain or order the dimensions alphabetically.
    domain_dims = [dim for dim in domain]
    spec_dims = set(lhs_dims + output_dims)
    assert set(domain_dims) == spec_dims, str.format(
      f"Expected domain dimensions to match specification {spec_dims}")

    self.__lhs_dims = lhs_dims
    self.__rhs_dims = rhs_dims
    self.__output_dims = output_dims
    self.__domain_dims = domain_dims

  @property
  def lhs_dims(self):
    """Dimensions of the LHS operand, as string and in order."""
    return self.__lhs_dims

  @property
  def rhs_dims(self):
    """Dimensions of the RHS operand, as string and in order, or None."""
    return self.__rhs_dims

  @property
  def output_dims(self):
    """Dimensions of the output tensors, as string and in order."""
    return self.__output_dims

  @property
  def domain_dims(self):
    """Dimensions of the iteration domain, as string and in order."""
    return self.__domain_dims

  @property
  def reduction_dims(self):
    """Reduction dimensions, as string and in order of LHS operand."""
    return "".join([d for d in self.lhs_dims if d not in self.output_dims])

  def __str__(self):
    """String representation of einsum."""
    operand_dims = [d for d in [self.lhs_dims, self.rhs_dims] if d is not None]
    return format(f"{','.join(operand_dims)}->{self.output_dims}")


def make_einsum(specification: EinsumSpecification):
  """Creates a Linalg structured op builder from einsum specification.

  The specification is similar to numpy.einsum and has the format:
  <lhs> ',' (<rhs>)? ('->' <out>)?`
  where <lhs>, <rhs>, and <out> are sequences of lowercase characters
  corresponding to indexes of the LHS, RHS, and output tensors, respectively.
  One-operand specifications define only the LHS dimensions. Two-operand
  specifications may omit the output dimensions. In this case, the output
  dimensions are inferred as the alphabetical ordered set of non-reduction
  LHS, RHS dimensions. For example, for the `nk,km` specification, the inferred
  output order is `mn` (rather than `nm` that would result from their
  concatenation).

  Only one-operand and two-operand specifications are currently supported, i.e.
  either a LHS or an LHS and a RHS tensor must be present, and the result must
  not have repeated dimensions.
  """
  lhs_dims = specification.lhs_dims
  rhs_dims = specification.rhs_dims
  output_dims = specification.output_dims
  reduction_dims = specification.reduction_dims
  domain_dims = specification.domain_dims

  def symbols(dimensions: str):
    """Return a tuple of OpDSL symbols that corresponds to dimensions."""
    return tuple(getattr(S, c.upper()) for c in dimensions)

  def dims(dimensions: str):
    """Return a tuple of OpDSL dimensions that corresponds to dimensions."""
    if not dimensions:
      return None
    return tuple(getattr(D, c) for c in dimensions)

  # Create and return a one-operand einsum operation.
  if rhs_dims is None:
    op_dims = "_".join([lhs_dims, output_dims])

    if reduction_dims:
      @linalg_structured_op(op_name="einsum_contraction_" + op_dims)
      def einsum_op(LHS=TensorDef(TV.T1, *symbols(lhs_dims)),
                    O=TensorDef(U, *symbols(output_dims), output=True)):
        domain(*dims(domain_dims))
        O[dims(output_dims)] += TypeFn.cast_signed(U, LHS[dims(lhs_dims)])
      return einsum_op
    else:
      @linalg_structured_op(op_name="einsum_transpose_" + op_dims)
      def einsum_op(LHS=TensorDef(U, *symbols(lhs_dims)),
                    O=TensorDef(U, *symbols(output_dims), output=True)):
        domain(*dims(domain_dims))
        O[dims(output_dims)] = TypeFn.cast_signed(U, LHS[dims(lhs_dims)])
      return einsum_op

  # Create and return a two-operand non-contraction operation.
  op_dims = "_".join([lhs_dims, rhs_dims, output_dims])

  if reduction_dims:
    @linalg_structured_op(op_name="einsum_contraction_" + op_dims)
    def einsum_op(LHS=TensorDef(TV.T1, *symbols(lhs_dims)),
                  RHS=TensorDef(TV.T2, *symbols(rhs_dims)),
                  O=TensorDef(U, *symbols(output_dims), output=True)):
      domain(*dims(domain_dims))
      implements(ContractionOpInterface)
      O[dims(output_dims)] += TypeFn.cast_signed(U,
          LHS[dims(lhs_dims)]) * TypeFn.cast_signed(U, RHS[dims(rhs_dims)])
    return einsum_op
  else:
    @linalg_structured_op(op_name="einsum_transpose_" + op_dims)
    def einsum_op(LHS=TensorDef(TV.T1, *symbols(lhs_dims)),
                  RHS=TensorDef(TV.T2, *symbols(rhs_dims)),
                  O=TensorDef(U, *symbols(output_dims), output=True)):
      domain(*dims(domain_dims))
      O[dims(output_dims)] = TypeFn.cast_signed(U,
          LHS[dims(lhs_dims)]) * TypeFn.cast_signed(U, RHS[dims(rhs_dims)])
    return einsum_op
