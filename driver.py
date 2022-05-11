from mlir.ir import Context, Module, Location
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager
import numpy as np
import ctypes


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
  llvm.func @_mlir__mlir_ciface_store(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: i64) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64)>
    %4 = llvm.mlir.constant(42 : i64) : i64
    %5 = llvm.extractvalue %3[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64)>
    llvm.store %4, %5 : !llvm.ptr<i64>
    llvm.return
  }
  llvm.func @_mlir_ciface__mlir__mlir_ciface_store(%arg0: !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64)>>) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64)>
    llvm.call @_mlir__mlir_ciface_store(%1, %2, %3) : (!llvm.ptr<i64>, !llvm.ptr<i64>, i64) -> ()
    llvm.return
  }
}
      """)
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
  exec_eng.invoke("_mlir__mlir_ciface_store", output_ptr)
  print(output)

# run configure with llvm path to proper llvm commit folder
# ninja install
# update PythonPath to
# /home/michel/opencompl/MLIR-lite/ChocoPyCompiler/:/home/michel/MasterThesis/iree-llvm-sandbox/build/tools/sandbox/python_packages/:/home/michel/opencompl/MLIR-lite/xdsl/src:/home/michel/MasterThesis/iree-llvm-sandbox/experimental/sql:/home/michel/opencompl/llvm-project/build/tools/mlir/python_packages/mlir_core
