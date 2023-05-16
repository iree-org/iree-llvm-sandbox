// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm \
// RUN: | FileCheck %s

// CHECK-LABEL: llvm.func @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<f32, 1>
// CHECK-SAME:      ) attributes {sym_visibility = "public"} {
// CHECK-NEXT:    llvm.return
tt.func public @kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  tt.return
}
