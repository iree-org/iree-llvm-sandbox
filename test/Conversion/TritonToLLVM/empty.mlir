// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm \
// RUN: | FileCheck %s

// CHECK-LABEL: llvm.func @kernel()
// CHECK-SAME:      attributes {sym_visibility = "public"} {
// CHECK-NEXT:    llvm.return
tt.func public @kernel() {
  tt.return
}
