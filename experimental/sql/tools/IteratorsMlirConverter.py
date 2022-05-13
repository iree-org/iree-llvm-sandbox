import mlir_iterators.ir
from xdsl import load_mlir_module

ir = mlir_iterators.ir
load_mlir_module(mlir_iterators)
from mlir_iterators.dialects import iterators as it
from xdsl.mlir_converter import MLIRConverter
from xdsl.ir import Attribute
from dialects.iterators import Stream
from importlib import import_module

# PYTHONPATH:
# /home/michel/opencompl/MLIR-lite/ChocoPyCompiler/:/home/michel/MasterThesis/iree-llvm-sandbox/build/tools/sandbox/python_packages/:/home/michel/opencompl/MLIR-lite/xdsl/src:/home/michel/MasterThesis/iree-llvm-sandbox/experimental/sql:/home/michel/opencompl/llvm-project/build/tools/mlir/python_packages/mlir_core


class IteratorsMlirConverter(MLIRConverter):

  def __init__(self, ctx):
    super().__init__(ctx)

  def register_external_dialects(self):
    it.register_dialect()

  def convert_type(self, typ: Attribute) -> ir.Type:
    if isinstance(typ, Stream):
      return it.StreamType.get(self.convert_type(typ.types))
    return super().convert_type(typ)
