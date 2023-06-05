// RUN: structured-opt %s \
// RUN:   -convert-triton-func-to-func \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func private @f()
tt.func private @f()

// CHECK-LABEL: func.func public @kernel() {
// CHECK-NEXT:    call @f() : () -> ()
// CHECK-NEXT:    return
tt.func public @kernel() {
  tt.call @f() : () -> ()
  tt.return
}
