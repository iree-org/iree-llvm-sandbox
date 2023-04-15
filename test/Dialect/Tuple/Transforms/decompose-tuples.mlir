// RUN: structured-opt %s -decompose-tuples \
// RUN: | FileCheck --enable-var-scope %s

// RUN: structured-opt %s -decompose-tuples="convert-scf-ops=false" \
// RUN: | FileCheck --enable-var-scope --check-prefix=CHECK-NO-SCF %s

!nested_tuple = tuple<tuple<>, i32, tuple<i64>>

// CHECK-LABEL: func.func @fromElements(
// CHECK-SAME:                          %[[ARG0:[^:]*]]: i32,
// CHECK-SAME:                          %[[ARG1:[^:]*]]: i64) -> (i32, i64) {
// CHECK-NEXT:    return %[[ARG0]], %[[ARG1]] : i32, i64
func.func @fromElements(%arg0 : tuple<>, %arg1 : i32, %arg2 : tuple<i64>) -> !nested_tuple {
  %tuple = tuple.from_elements %arg0, %arg1, %arg2 : !nested_tuple
  return %tuple : !nested_tuple
}

// CHECK-LABEL: func.func @toElements(
// CHECK-SAME:                        %[[ARG0:[^:]*]]: i32,
// CHECK-SAME:                        %[[ARG1:[^:]*]]: i64) -> (i32, i64) {
// CHECK-NEXT:    return %[[ARG0]], %[[ARG1]] : i32, i64
func.func @toElements(%input : !nested_tuple) -> (tuple<>, i32, tuple<i64>) {
  %elements:3 = tuple.to_elements %input : !nested_tuple
  return %elements#0, %elements#1, %elements#2 : tuple<>, i32, tuple<i64>
}

// CHECK-LABEL: func.func @argumentMaterialization(
// CHECK-SAME:                                     %[[ARG0:[^:]*]]: i32,
// CHECK-SAME:                                     %[[ARG1:[^:]*]]: i64) -> i32 {
// CHECK-DAG:     %[[V0:.*]] = tuple.from_elements  : tuple<>
// CHECK-DAG:     %[[V1:.*]] = tuple.from_elements %[[ARG1]] : tuple<i64>
// CHECK-NEXT:    %[[V2:.*]] = tuple.from_elements %[[V0]], %[[ARG0]], %[[V1]] : tuple<tuple<>, i32, tuple<i64>>
// CHECK-NEXT:    %[[V3:.*]] = builtin.unrealized_conversion_cast %[[V2]] : tuple<tuple<>, i32, tuple<i64>> to i32
func.func @argumentMaterialization(%input : !nested_tuple) -> i32 {
  %0 = builtin.unrealized_conversion_cast %input : !nested_tuple to i32
  return %0 : i32
}

// CHECK-LABEL: func.func @sourceMaterialization() -> i32 {
// CHECK-DAG:     %[[V0:.*]] = arith.constant 1 : i32
// CHECK-DAG:     %[[V1:.*]] = arith.constant 2 : i64
// CHECK-DAG:     %[[V2:.*]] = tuple.from_elements  : tuple<>
// CHECK-DAG:     %[[V3:.*]] = tuple.from_elements %[[V1]] : tuple<i64>
// CHECK-NEXT:    %[[V4:.*]] = tuple.from_elements %[[V2]], %[[V0]], %[[V3]] : tuple<tuple<>, i32, tuple<i64>>
// CHECK-NEXT:    %[[V5:.*]] = builtin.unrealized_conversion_cast %[[V4]] : tuple<tuple<>, i32, tuple<i64>> to i32
func.func @sourceMaterialization() -> i32 {
  %i32 = arith.constant 1 : i32
  %i64 = arith.constant 2 : i64
  %empty_tuple = tuple.from_elements : tuple<>
  %i64_tuple = tuple.from_elements %i64 : tuple<i64>
  %tuple = tuple.from_elements %empty_tuple, %i32, %i64_tuple : !nested_tuple
  %0 = builtin.unrealized_conversion_cast %tuple : !nested_tuple to i32
  return %0 : i32
}

// CHECK-LABEL: func.func @targetMaterialization() -> (i32, i64) {
// CHECK-DAG:     %[[V0:.*]] = builtin.unrealized_conversion_cast to tuple<tuple<>, i32, tuple<i64>>
// CHECK-DAG:     %[[V1:.*]]:3 = tuple.to_elements %[[V0]] : tuple<tuple<>, i32, tuple<i64>>
// CHECK-DAG:     tuple.to_elements %[[V1]]#0 : tuple<>
// CHECK-DAG:     %[[V2:.*]] = tuple.to_elements %elements#2 : tuple<i64>
// CHECK-NEXT:    return %[[V1]]#1, %[[V2]] : i32, i64
func.func @targetMaterialization() -> !nested_tuple {
  %tuple = builtin.unrealized_conversion_cast to !nested_tuple
  return %tuple : !nested_tuple
}

// CHECK-LABEL: func.func @scfIf(
// CHECK-SAME:                   %[[ARG0:[^:]*]]: i1) -> (i32, i64) {
// CHECK:         %[[V0:.*]]:2 = scf.if %[[ARG0]] -> (i32, i64) {
// CHECK-NEXT:      scf.yield %[[V1:.*]], %[[V2:.*]] : i32, i64
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %[[V3:.*]], %[[V4:.*]] : i32, i64
// CHECK-NEXT:    }
// CHECK-NO-SCF-LABEL: func.func @scfIf(
// CHECK-NO-SCF-SAME:                   %[[ARG0:[^:]*]]: i1) -> (i32, i64) {
// CHECK-NO-SCF:         %[[V0:.*]] = scf.if %[[ARG0]] -> (tuple<tuple<>, i32, tuple<i64>>) {
// CHECK-NO-SCF:         %[[V1:.*]]:3 = tuple.to_elements %[[V0]] : tuple<tuple<>, i32, tuple<i64>>
func.func @scfIf(%cmp : i1) -> !nested_tuple {
  %result = scf.if %cmp -> !nested_tuple {
    %i32 = arith.constant 1 : i32
    %i64 = arith.constant 2 : i64
    %empty_tuple = tuple.from_elements : tuple<>
    %i64_tuple = tuple.from_elements %i64 : tuple<i64>
    %tuple = tuple.from_elements %empty_tuple, %i32, %i64_tuple : !nested_tuple
    scf.yield %tuple : !nested_tuple
  } else {
    %i32 = arith.constant -1 : i32
    %i64 = arith.constant -2 : i64
    %empty_tuple = tuple.from_elements : tuple<>
    %i64_tuple = tuple.from_elements %i64 : tuple<i64>
    %tuple = tuple.from_elements %empty_tuple, %i32, %i64_tuple : !nested_tuple
    scf.yield %tuple : !nested_tuple
  }
  return %result : !nested_tuple
}
