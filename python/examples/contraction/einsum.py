from mlir.ir import *
from mlir.dialects.linalg.opdsl.lang import *

import itertools


class EinsumSpecification:
  """Structured representation of a string einsum specification."""

  def __init__(self, specification: str):
    """Creates a specification given its string format."""
    # Split out the operands and the result part of the specification.
    specification_split = specification.split("->")
    assert len(specification_split
              ) <= 2, "Expected at most one '->' in the specification."
    operands = specification_split[0]
    result = specification_split[1] if len(specification_split) > 1 else None

    # Split LHS and RHS operands.
    operands_split = operands.split(",")
    assert len(
        operands_split
    ) <= 2, "Expected at most two comma-separated inputs in the specification."
    if len(operands_split) != 2:
      raise NotImplementedError("single-argument einsum")

    # Reduction dimensions are the symbols common to LHS and RHS. Output
    # dimensions are the symbols unique to LHS and RHS, in alphabetical order.
    lhs_dims, rhs_dims = operands_split
    lhs_dims = lhs_dims.strip()
    rhs_dims = rhs_dims.strip()
    assert len(set(lhs_dims)) == len(
        lhs_dims), "Unexpected duplicate symbol on the LHS."
    assert lhs_dims.islower(), "Expected only lowercase symbols on the LHS."
    assert len(set(rhs_dims)) == len(
        rhs_dims), "Unexpected duplicate symbol on the RHS."
    assert rhs_dims.islower(), "Expected only lowercase symbols on the RHS."
    reduction_dims = "".join([d for d in lhs_dims if d in rhs_dims])
    inferred_output_dims = sorted([
        d for d in itertools.chain(lhs_dims, rhs_dims)
        if d not in reduction_dims
    ])

    # Use the inferred output specification if there is no user-specified one.
    if result is not None:
      result = result.strip()
      for d in result:
        assert d in inferred_output_dims, "Undefined dimension in the result."
      output_dims = result
      if len(set(output_dims)) != len(output_dims):
        raise NotImplementedError("Repeated output dimensions.")
    else:
      output_dims = "".join(inferred_output_dims)

    self.__lhs_dims = lhs_dims
    self.__rhs_dims = rhs_dims
    self.__output_dims = output_dims

  @property
  def lhs_dims(self):
    """Dimensions of the LHS operand, as string and in order."""
    return self.__lhs_dims

  @property
  def rhs_dims(self):
    """Dimensions of the RHS operand, as string and in order."""
    return self.__rhs_dims

  @property
  def output_dims(self):
    """Dimensions of the output tensors, as string and in order."""
    return self.__output_dims

  @property
  def reduction_dims(self):
    """Reduction dimensions, as string and in order of LHS operand."""
    return "".join([d for d in self.lhs_dims if d in self.rhs_dims])

  def __str__(self):
    """String representation of einsum."""
    return format(f"{self.lhs_dims},{self.rhs_dims}->{self.output_dims}")


def make_einsum(specification: str):
  """Creates a Linalg structured op builder from einsum specificaiton.

  The specification is similar to that of numpy.einsum: `lhs,rhs->output`,
  where lhs,rhs,output are sequences of lowercase characters corresponding to
  indexes of the LHS, RHS and output tensors, respectively. The characters that
  appear in both LHS and RHS correspond to reduction (contraction) dimensions
  and must not be present in the output specification.

  The output `->output` part of the specification may be omitted. In this case,
  it is inferred as the alphabetical order of non-reduction LHS, RHS dimensions.
  For example, for the `nk,km` specification, the inferred output order is `mn`
  (rather than `nm` that would result from their concatenation).

  Only 2-operand contractions are currently supported, i.e. both LHS and RHS
  tensors must be present, and the result must not have repeated dimensions.
  """
  spec = EinsumSpecification(specification)
  lhs_dims = spec.lhs_dims
  rhs_dims = spec.rhs_dims
  output_dims = spec.output_dims
  reduction_dims = spec.reduction_dims

  def symbols(dimensions: str):
    """Return a tuple of OpDSL symbols that corresponds to dimensions."""
    return tuple(getattr(S, c.upper()) for c in dimensions)

  def dims(dimensions: str):
    """Return a tuple of OpDSL dimensions that corresponds to dimensions."""
    return tuple(getattr(D, c) for c in dimensions)

  # Create and return the contraction structured op.
  op_name = "_".join(["contraction", lhs_dims, rhs_dims, output_dims])

  @linalg_structured_op(op_name=op_name)
  def einsum_contraction(LHS=TensorDef(TV.T1, *symbols(lhs_dims)),
                         RHS=TensorDef(TV.T2, *symbols(rhs_dims)),
                         O=TensorDef(U, *symbols(output_dims), output=True)):
    domain(*dims(output_dims + reduction_dims))
    implements(ContractionOpInterface)
    O[dims(output_dims)] += TypeFn.cast(U, LHS[dims(lhs_dims)]) * TypeFn.cast(
        U, RHS[dims(rhs_dims)])

  return einsum_contraction
