import mlir_iterators.ir as ir
from mlir_iterators.dialects import iterators as it
from xdsl.mlir_converter import MLIRConverter
from xdsl.ir import Attribute
from dialects.iterators import Stream, ColumnarBatch

# This MLIRConverter enriches the standard xdsl MLIRConverter
# (https://github.com/xdslproject/xdsl/blob/main/src/xdsl/mlir_converter.py) by
# adding the types of the iterators dialect
# (https://github.com/google/iree-llvm-sandbox/tree/main/experimental/iterators).


class IteratorsMLIRConverter(MLIRConverter):

  def register_external_dialects(self):
    it.register_dialect()
    super().register_external_dialects()

  def convert_type(self, typ: Attribute) -> ir.Type:
    if isinstance(typ, Stream):
      return it.StreamType.get(self.convert_type(typ.types))
    if isinstance(typ, ColumnarBatch):
      return it.ColumnarBatchType.get(self.convert_type(typ.elementType))
    return super().convert_type(typ)
