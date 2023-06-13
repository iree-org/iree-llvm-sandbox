// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<f32, 1>
// CHECK-NEXT:    return
func.func public @kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  return
}
