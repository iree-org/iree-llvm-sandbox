// RUN: mlir-proto-opt %s \
// RUN: | FileCheck %s

func.func @main(%input : !tabular.tabular_view<i32>) {
  // CHECK-LABEL: func.func @main(%{{arg.*}}: !tabular.tabular_view<i32>) {
  %stream = iterators.tabular_view_to_stream %input
                to !iterators.stream<!llvm.struct<(i32)>>
// CHECK-NEXT:    %[[V0:.*]] = iterators.tabular_view_to_stream %[[arg0:.*]] to !iterators.stream<!llvm.struct<(i32)>>
  return
// CHECK-NEXT:    return
}
// CHECK-NEXT:  }
