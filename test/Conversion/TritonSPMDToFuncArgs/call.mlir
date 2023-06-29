// RUN: structured-opt %s \
// RUN:   -convert-triton-spmd-to-func-args \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @func_f(
// CHECK-SAME:      %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32,
// CHECK-SAME:      %[[ARG6:.*]]: f32) -> f32 {
// CHECK-NEXT:    %[[V0:.*]] = arith.addf %[[ARG6]], %[[ARG6]] : f32
// CHECK-NEXT:    return %[[V0]] : f32
func.func public @func_f(%arg0: f32) -> f32 {
  %0 = arith.addf %arg0, %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: tt.func public @tt_f(
// CHECK-SAME:      %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32,
// CHECK-SAME:      %[[ARG6:.*]]: f32) -> f32 {
// CHECK-NEXT:    %[[V0:.*]] = arith.addf %[[ARG6]], %[[ARG6]] : f32
// CHECK-NEXT:    tt.return %[[V0]] : f32
tt.func public @tt_f(%arg0: f32) -> f32 {
  %0 = arith.addf %arg0, %arg0 : f32
  tt.return %0 : f32
}

// CHECK-LABEL: func.func public @kernel_f2f(
// CHECK-SAME:      %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32,
// CHECK-SAME:      %[[ARG6:.*]]: f32) -> f32 {
// CHECK-NEXT:    %[[V0:.*]] = call @func_f(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]]) : (i32, i32, i32, i32, i32, i32, f32) -> f32
// CHECK-NEXT:    return %[[V0]] : f32
func.func public @kernel_f2f(%arg0: f32) -> f32 {
  %0 = func.call @func_f(%arg0) : (f32) -> f32
  return %0 : f32
}

// CHECK-LABEL: func.func public @kernel_f2t(
// CHECK-SAME:      %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32,
// CHECK-SAME:      %[[ARG6:.*]]: f32) -> f32 {
// CHECK-NEXT:    %[[V0:.*]] = tt.call @tt_f(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]]) : (i32, i32, i32, i32, i32, i32, f32) -> f32
// CHECK-NEXT:    return %[[V0]] : f32
func.func public @kernel_f2t(%arg0: f32) -> f32 {
  %0 = tt.call @tt_f(%arg0) : (f32) -> f32
  return %0 : f32
}

// CHECK-LABEL: tt.func public @kernel_t2f(
// CHECK-SAME:      %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32,
// CHECK-SAME:      %[[ARG6:.*]]: f32) -> f32 {
// CHECK-NEXT:    %[[V0:.*]] = func.call @func_f(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]]) : (i32, i32, i32, i32, i32, i32, f32) -> f32
// CHECK-NEXT:    tt.return %[[V0]] : f32
tt.func public @kernel_t2f(%arg0: f32) -> f32 {
  %0 = func.call @func_f(%arg0) : (f32) -> f32
  tt.return %0 : f32
}

// CHECK-LABEL: tt.func public @t2t(
// CHECK-SAME:      %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32,
// CHECK-SAME:      %[[ARG6:.*]]: f32) -> f32 {
// CHECK-NEXT:    %[[V0:.*]] = tt.call @tt_f(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]]) : (i32, i32, i32, i32, i32, i32, f32) -> f32
// CHECK-NEXT:    tt.return %[[V0]] : f32
tt.func public @t2t(%arg0: f32) -> f32 {
  %0 = tt.call @tt_f(%arg0) : (f32) -> f32
  tt.return %0 : f32
}
