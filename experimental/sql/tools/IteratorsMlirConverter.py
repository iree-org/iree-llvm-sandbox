import mlir_iterators.ir as ir
from mlir_iterators.dialects import iterators as it
from xdsl.mlir_converter import MLIRConverter
from xdsl.ir import Attribute
from dialects.iterators import Stream


class IteratorsMlirConverter(MLIRConverter):

  def register_external_dialects(self):
    it.register_dialect()
    super().register_external_dialects()

  def convert_type(self, typ: Attribute) -> ir.Type:
    if isinstance(typ, Stream):
      return it.StreamType.get(self.convert_type(typ.types))
    return super().convert_type(typ)
