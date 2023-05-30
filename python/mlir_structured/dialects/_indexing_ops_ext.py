#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
  from .. import ir
  from ._ods_common import (
      get_op_result_or_value as _get_op_result_or_value,
      get_op_results_or_values as _get_op_results_or_values,
      get_default_loc_context as _get_default_loc_context,
  )
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e


class ARangeOp:
  OPERATION_NAME = "indexing.arange"

  def __init__(self,
               *,
               start=None,
               stop=None,
               step=None,
               fold=None,
               loc=None,
               ip=None):
    operands = []

    for i, inp in enumerate([start, stop, step]):
      if not bool(isinstance(inp, int)) != bool(isinstance(inp, ir.Value)):
        raise RuntimeError(
            f"expected either Value or int but not both for arg {i=}: {inp}.")

    attributes = {"operand_segment_sizes": []}
    if start is not None and isinstance(start, ir.Value):
      operands.append(_get_op_result_or_value(start))
      attributes["operand_segment_sizes"].append(1)
      startAttr = None
    else:
      attributes["operand_segment_sizes"].append(0)
      startAttr = start

    if stop is not None and isinstance(stop, ir.Value):
      operands.append(_get_op_result_or_value(stop))
      attributes["operand_segment_sizes"].append(1)
      stopAttr = None
    else:
      attributes["operand_segment_sizes"].append(0)
      stopAttr = stop

    if step is not None and isinstance(step, ir.Value):
      operands.append(_get_op_result_or_value(step))
      attributes["operand_segment_sizes"].append(1)
      stepAttr = None
    else:
      attributes["operand_segment_sizes"].append(0)
      stepAttr = step

    attributes["operand_segment_sizes"] = ir.DenseI32ArrayAttr.get(
        attributes["operand_segment_sizes"])

    _ods_context = _get_default_loc_context(loc)

    if startAttr is not None:
      attributes["startAttr"] = (startAttr if
                                 (issubclass(type(startAttr), ir.Attribute) or
                                  not ir.AttrBuilder.contains('IndexAttr')) else
                                 ir.AttrBuilder.get('IndexAttr')(
                                     startAttr, context=_ods_context))
    if stopAttr is not None:
      attributes["stopAttr"] = (stopAttr if
                                (issubclass(type(stopAttr), ir.Attribute) or
                                 not ir.AttrBuilder.contains('IndexAttr')) else
                                ir.AttrBuilder.get('IndexAttr')(
                                    stopAttr, context=_ods_context))
    if stepAttr is not None:
      attributes["stepAttr"] = (stepAttr if
                                (issubclass(type(stepAttr), ir.Attribute) or
                                 not ir.AttrBuilder.contains('IndexAttr')) else
                                ir.AttrBuilder.get('IndexAttr')(
                                    stepAttr, context=_ods_context))
    if fold is not None:
      attributes["foldAttr"] = (fold if
                                (issubclass(type(fold), ir.Attribute) or
                                 not ir.AttrBuilder.contains('BoolAttr')) else
                                ir.AttrBuilder.get('BoolAttr')(
                                    fold, context=_ods_context))

    results = ir.InferTypeOpInterface(ARangeOp).inferReturnTypes(
        operands=operands,
        attributes=ir.DictAttr.get(attributes, context=_ods_context),
        context=_ods_context,
        loc=loc)
    _ods_successors = None
    # TODO(max): use a regular builder after https://reviews.llvm.org/D151409
    # gets merged and we catch up.
    arange_op = ir.Operation.create(self.OPERATION_NAME,
                                    operands=operands,
                                    results=results,
                                    regions=0,
                                    attributes=attributes,
                                    loc=loc,
                                    ip=ip)
    ir.OpView.__init__(self, arange_op)


def get_gather_result_shape(source, indices, gather_dims):
  from ._indexing_ops_gen import GatherOp

  attributes = {
      "gather_dims":
          ir.AttrBuilder.get('DenseI64ArrayAttr')(gather_dims, context=None)
  }
  results = ir.InferTypeOpInterface(GatherOp).inferReturnTypes(
      operands=[source, indices], attributes=ir.DictAttr.get(attributes))

  assert len(results) == 1
  return tuple(ir.RankedTensorType(results[0]).shape)
