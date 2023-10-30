from ._jasc_transform_ops_gen import *
from ._jasc_transform_ops_gen import _Dialect
from ..._mlir_libs._mlirTransformOpsJasc import *

try:
  from typing import Sequence
  from jaxlib.mlir import ir
  from jaxlib.mlir.dialects import pdl
  from ._ods_common import _cext as _ods_cext

except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e


@_ods_cext.register_operation(_Dialect, replace=True)
class MatchTagOp(MatchTagOp):
  """Specialization for the MatchTag op class."""

  def __init__(
      self,
      target: ir.Value,
      tags: Sequence[str],
      *,
      ip=None,
      loc=None,
  ):
    result_ty = pdl.OperationType.get()
    tags_attr = ir.ArrayAttr.get(list(map(ir.StringAttr.get, tags)))
    super().__init__(result_ty, target, tags_attr, ip=ip, loc=loc)
