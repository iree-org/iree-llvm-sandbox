from typing import Callable, Sequence

import mlir.dialects.pdl as pdl
import mlir.ir as ir


def make_single_op_pdl_pattern(
    module,
    pdl_pattern_name: str,
    op_to_match_name: str,
    constraints_builder_list: Sequence[Callable[
        [pdl.OperationOp, pdl.OperandsOp, pdl.TypesOp], None]],
    benefit: int = 1):
  """Given a module, create a new pdl.PatternOp @ module.body insertion point.
  
  The pdl.PatternOp is additionally populated with constraints created from the
  constraints_builder_list. Each such builder takes as arguments:
    - the pdl.OperationOp of `op_to_match_name` that we are trying to match
    - the pdl.OperandsOp that captures the ValueRange with the op operands
    - the pdl.TypesOp that captures the TypeRange with the op result types

  Each such constraints become a native pdl.ApplyNativeConstraintOp that calls
  into the proper C++ implementation.
  """

  with ir.InsertionPoint(module.body):
    pdl_pattern = pdl.PatternOp(benefit=benefit, name=pdl_pattern_name)
    with ir.InsertionPoint(pdl_pattern.body):
      operands = pdl.OperandsOp()
      result_types = pdl.TypesOp()
      pdl_op = pdl.OperationOp(op_to_match_name,
                               args=[operands],
                               types=[result_types])
      for constraints_builder in constraints_builder_list:
        constraints_builder(pdl_op, operands, result_types)
      # TODO: we don't want this, but it is the required terminator for pdl.pattern
      pdl.RewriteOp(pdl_op, 'linalg_transform.apply')


###############################################################################
# Constraints (TODO: switch to wrapping PDLL and searching in python)
###############################################################################
def i64_attr(val: int):
  return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), val)


def constraint_is_operand_dim_multiple_of(operands: pdl.OperandsOp,
                                          operand_number: int, dim: int,
                                          divisor: int):
  """Assume this is called under `with ir.InsertionPoint(pdl_pattern.body)`.

  Create a pdl.ApplyNativeConstraintOp that filters whether operands[operand_number]
  is a ShapedType whose dimension `dim` is a multiple of `divisor`.

  The following conventions are used:
    * `divisor == 0` is considered to divide any size, including dynamic sizes (i.e. -1)
    * `divisor == 1` is considered to divide any static size, excluding dynamic sizes
    * `divisor  > 1` divides only static sizes `k * divisor`, where `k > 0`
  """

  return pdl.ApplyNativeConstraintOp(\
    "isDimMultipleOf",
    args=[operands],
    params=[ir.DictAttr.get({'operand_number': i64_attr(operand_number),
                             'dim': i64_attr(dim),
                             'divisor': i64_attr(divisor),})]
    )


def constraint_is_operand_dim_dynamic(operands: pdl.OperandsOp,
                                      operand_number: int, dim: int):
  """Assume this is called under `with ir.InsertionPoint(pdl_pattern.body)`.

  Create a pdl.ApplyNativeConstraintOp that filters whether operands[operand_number]
  is a ShapedType whose dimension `dim` is dynamic.
  """

  return pdl.ApplyNativeConstraintOp(\
    "isDimDynamic",
    args=[operands],
    params=[ir.DictAttr.get({'operand_number': i64_attr(operand_number),
                             'dim': i64_attr(dim),})]
    )


def constraint_is_operand_dim_static(operands: pdl.OperandsOp,
                                     operand_number: int, dim: int):
  """Assume this is called under `with ir.InsertionPoint(pdl_pattern.body)`.

  Create a pdl.ApplyNativeConstraintOp that filters whether operands[operand_number]
  is a ShapedType whose dimension `dim` is static.
  """
  return pdl.ApplyNativeConstraintOp(\
    "isDimStatic",
    args=[operands],
    params=[ir.DictAttr.get({'operand_number': i64_attr(operand_number),
                             'dim': i64_attr(dim),})]
    )


def constraint_is_equivalent_to_op(pdl_op: pdl.OperationOp,
                                   desired_op_name: str):
  """Assume this is called under `with ir.InsertionPoint(pdl_pattern.body)`.

  Create a pdl.ApplyNativeConstraintOp that filters whether the Operation* 
  captured by pdl_op is equivalent to `desired_op_name`.
  This is mostly used for linalg ops.
  The underlying implementation 
  """

  return pdl.ApplyNativeConstraintOp(
      "isEquivalentToOp",
      args=[pdl_op],
      params=[ir.StringAttr.get(desired_op_name)])


###############################################################################
# Constraint builders make it a nicer API but only wrap the constraint
###############################################################################


def make_constraint_operand_dim_divisible_by(operand_number: int, dim: int,
                                             divisor: int):

  def constraints_builder(pdl_op: pdl.OperationOp, operands: pdl.OperandsOp,
                          _: pdl.TypesOp):
    """Assume this is called under `with ir.InsertionPoint(pdl_pattern.body)`."""
    constraint_is_operand_dim_multiple_of(operands=operands,
                                          operand_number=operand_number,
                                          dim=dim,
                                          divisor=divisor)

  return constraints_builder


def make_constraint_operand_dim_dynamic(operand_number: int, dim: int):

  def constraints_builder(pdl_op: pdl.OperationOp, operands: pdl.OperandsOp,
                          _: pdl.TypesOp):
    """Assume this is called under `with ir.InsertionPoint(pdl_pattern.body)`."""
    constraint_is_operand_dim_dynamic(
        operands=operands,
        operand_number=operand_number,
        dim=dim,
    )

  return constraints_builder


def make_constraint_operand_dim_static(operand_number: int, dim: int):

  def constraints_builder(pdl_op: pdl.OperationOp, operands: pdl.OperandsOp,
                          _: pdl.TypesOp):
    """Assume this is called under `with ir.InsertionPoint(pdl_pattern.body)`."""
    constraint_is_operand_dim_static(
        operands=operands,
        operand_number=operand_number,
        dim=dim,
    )

  return constraints_builder


def make_constraint_is_equivalent_to_op(op_name: str):

  def constraints_builder(pdl_op: pdl.OperationOp, operands: pdl.OperandsOp,
                          _: pdl.TypesOp):
    """Assume this is called under `with ir.InsertionPoint(pdl_pattern.body)`."""
    constraint_is_equivalent_to_op(pdl_op, op_name)

  return constraints_builder


###############################################################################
# More advanced constraint builders.
###############################################################################


def match_op_with_sizes_multiple_of(
    module,
    equivalent_op_name: str,
    divisors_list: Sequence[int] = [],
    op_dim_spec_list: Sequence[Sequence[int]] = [],
    op_to_match_name: str = 'linalg.generic'):
  """
  Return a constraint builder that matches `op_to_match_name` that are known to 
  be equivalent to `equivalent_op_name` and such that a subset of dimensions 
  are constrained to be multiples of fizes passed in `divisors_list`.

  Parameters:
  * divisors_list: list of integers, one for each entry in `op_dim_spec_list`
  * op_dim_spec_list: list of pairs of (operand number, dim) for which static
    or dynamic constraint will be added according to the corresponding entry in
    `dynamic_spec_list`.

  The following conventions are used:
    * `divisor == 0` is considered to divide any size, including dynamic sizes (i.e. -1)
    * `divisor == 1` is considered to divide any static size, excluding dynamic sizes
    * `divisor  > 1` divides only static sizes `k * divisor`, where `k > 0`

  Note: dynamic_spec_list and op_dim_spec_list must have the same length
  """

  assert len(divisors_list) == len(
      op_dim_spec_list
  ), 'divisors_list and op_dim_spec_list must be of the same length'
  constraints_builder_list = [
      make_constraint_is_equivalent_to_op(equivalent_op_name)
  ]

  match_name = 'isa_' + equivalent_op_name
  for sz, op_dim in zip(divisors_list, op_dim_spec_list):
    constraints_builder_list.append(
        make_constraint_operand_dim_divisible_by(operand_number=op_dim[0],
                                                 dim=op_dim[1],
                                                 divisor=sz))
    match_name = match_name + '_x' + str(sz)

  make_single_op_pdl_pattern(module,
                             match_name,
                             op_to_match_name,
                             constraints_builder_list=constraints_builder_list)
  return match_name


def match_op_with_dynamic_or_static_sizes(
    module,
    equivalent_op_name: str,
    dynamic_spec_list: Sequence[str] = [],
    op_dim_spec_list: Sequence[Sequence[int]] = [],
    op_to_match_name: str = 'linalg.generic'):
  """
  Return a constraint builder that matches `op_to_match_name` that are known to 
  be equivalent to `equivalent_op_name` and such that a subset of dimensions 
  are constrained to be static or dynamic according to `dynamic_spec_list`.

  Parameters:
  * dynamic_spec_list: list of 's' or 'd' characters, one for each entry in 
    `op_dim_spec_list`
  * op_dim_spec_list: list of pairs of (operand number, dim) for which static
    or dynamic constraint will be added according to the corresponding entry in
    `dynamic_spec_list`.

  Note: dynamic_spec_list and op_dim_spec_list must have the same length
  """

  assert len(dynamic_spec_list) == len(
      op_dim_spec_list
  ), 'divisors_list and op_dim_spec_list must be of the same length'

  constraints_builder_list = [
      make_constraint_is_equivalent_to_op(equivalent_op_name)
  ]
  match_name = 'isa_' + equivalent_op_name
  for sp, op_dim in zip(dynamic_spec_list, op_dim_spec_list):
    if sp == 's':
      constraints_builder_list.append(
          make_constraint_operand_dim_static(operand_number=op_dim[0],
                                             dim=op_dim[1]))
      match_name = match_name + '_s'
    elif sp == 'd':
      constraints_builder_list.append(
          make_constraint_operand_dim_dynamic(operand_number=op_dim[0],
                                              dim=op_dim[1]))
      match_name = match_name + '_d'
    else:
      raise Exception(f'Not a valid static or dynamic specifier: {sp}')

  make_single_op_pdl_pattern(module,
                             match_name,
                             op_to_match_name,
                             constraints_builder_list=constraints_builder_list)
  return match_name