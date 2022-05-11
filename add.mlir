func.func @_mlir__mlir_ciface_store(%arg1 : memref<i64>) attributes { llvm.emit_c_interface} {
  %c0 = arith.constant 0 : i64
  memref.store %c0, %arg1[] : memref<i64>
  return
}
