from mlir.ir import Context, Module, Location
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager
import numpy as np
import ctypes
from mlir.conversions import _cextConversions
import mlir.dialects.memref as memref


def make_zero_d_memref_descriptor(dtype):

  class MemRefDescriptor(ctypes.Structure):
    """
        Build an empty descriptor for the given dtype, where rank=0.
        """

    _fields_ = [
        ("allocated", ctypes.c_longlong),
        ("aligned", ctypes.POINTER(dtype)),
        ("offset", ctypes.c_longlong),
    ]

  return MemRefDescriptor


with Context() as ctx, Location.unknown() as loc:

  mod = Module.parse("""
module attributes {llvm.data_layout = ""} {
func.func @store(%arg1 : memref<i64>) attributes { llvm.emit_c_interface} {
  %c0 = arith.constant 42 : i64
  memref.store %c0, %arg1[] : memref<i64>
  return
  }
}
      """)
  print(mod)

  PassManager.parse('convert-memref-to-llvm').run(mod)
  PassManager.parse('convert-func-to-llvm').run(mod)
  PassManager.parse('reconcile-unrealized-casts').run(mod)
  print(mod)

  shared_libs = ["build/lib/libruntime_utils.so"]

  output = np.array(1)

  def make_memref_from_arr(nparray):
    x = make_zero_d_memref_descriptor(np.ctypeslib.as_ctypes_type(
        nparray.dtype))()
    x.allocated = nparray.ctypes.data
    x.aligned = nparray.ctypes.data_as(
        ctypes.POINTER(np.ctypeslib.as_ctypes_type(nparray.dtype)))
    x.offset = ctypes.c_longlong(0)
    return x

  output_memref = make_memref_from_arr(output)

  output_ptr = ctypes.pointer(ctypes.pointer(output_memref))

  exec_eng = ExecutionEngine(mod, shared_libs=shared_libs)
  exec_eng.invoke("store", output_ptr)
  print(output)

# run configure with llvm path to proper llvm commit folder
# ninja install
# update PythonPath to
# /home/michel/opencompl/MLIR-lite/ChocoPyCompiler/:/home/michel/MasterThesis/iree-llvm-sandbox/build/tools/sandbox/python_packages/:/home/michel/opencompl/MLIR-lite/xdsl/src:/home/michel/MasterThesis/iree-llvm-sandbox/experimental/sql:/home/michel/opencompl/llvm-project/build/tools/mlir/python_packages/mlir_core
