// RUN: iterators-opt %s -convert-iterators-to-llvm \
// RUN: | FileCheck --enable-var-scope %s

func.func private @sum_tuple(
      %acc : tuple<i32>, %val : tuple<i32>) -> tuple<i32> {
  %acci = tuple.to_elements %acc : tuple<i32>
  %vali = tuple.to_elements %val : tuple<i32>
  %i = arith.addi %acci, %vali : i32
  %result = tuple.from_elements %i : tuple<i32>
  return %result : tuple<i32>
}

// CHECK-LABEL: func.func private @iterators.accumulate.close.{{[0-9]+}}(
// CHECK-SAME:        %[[ARG0:.*]]: !iterators.state<[[upstreamStateType:!iterators.state<[^>]*>]], tuple<i32>, i1>) ->
// CHECK-SAME:            !iterators.state<[[upstreamStateType]], tuple<i32>, i1> {
// CHECK-NEXT:    %[[V0:.*]] = iterators.extractvalue %[[ARG0]][0] : !iterators.state<[[upstreamStateType]], tuple<i32>, i1>
// CHECK-NEXT:    %[[V1:.*]] = call @iterators.{{.*}}.close.{{.*}}(%[[V0]]) : ([[upstreamStateType]]) -> [[upstreamStateType]]
// CHECK-NEXT:    %[[V2:.*]] = iterators.insertvalue %[[V1]] into %[[ARG0]][0] : !iterators.state<[[upstreamStateType]], tuple<i32>, i1>
// CHECK-NEXT:    return %[[V2]] : !iterators.state<[[upstreamStateType]], tuple<i32>, i1>

// CHECK-LABEL: func.func private @iterators.accumulate.next.{{[0-9]+}}(
// CHECK-SAME:        %[[ARG0:.*]]: !iterators.state<[[upstreamStateType:!iterators.state<[^>]*>]], tuple<i32>, i1>) ->
// CHECK-SAME:            (!iterators.state<[[upstreamStateType]], tuple<i32>, i1>, i1, tuple<i32>) {
// CHECK-NEXT:    %[[V1:.*]] = iterators.extractvalue %[[ARG0]][0] : !iterators.state<[[upstreamStateType]], tuple<i32>, i1>
// CHECK-NEXT:    %[[V2:.*]] = iterators.extractvalue %[[ARG0]][1] : !iterators.state<[[upstreamStateType]], tuple<i32>, i1>
// CHECK-NEXT:    %[[V3:.*]] = iterators.extractvalue %[[ARG0]][2] : !iterators.state<[[upstreamStateType]], tuple<i32>, i1>
// CHECK-NEXT:    %[[V4:.*]]:2 = scf.if %[[V3]] -> ([[upstreamStateType]], tuple<i32>) {
// CHECK-NEXT:      scf.yield %[[V1]], %[[V2]] : [[upstreamStateType]], tuple<i32>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[V5:.*]]:3 = scf.while (%[[arg1:.*]] = %[[V1]], %[[arg2:.*]] = %[[V2]]) : ([[upstreamStateType]], tuple<i32>) -> ([[upstreamStateType]], tuple<i32>, tuple<i32>) {
// CHECK-NEXT:        %[[V6:.*]]:3 = func.call @iterators.{{.*}}.next.{{.*}}(%[[arg1]]) : ([[upstreamStateType]]) -> ([[upstreamStateType]], i1, tuple<i32>)
// CHECK-NEXT:        scf.condition(%[[V6]]#1) %[[V6]]#0, %[[arg2]], %[[V6]]#2 : [[upstreamStateType]], tuple<i32>, tuple<i32>
// CHECK-NEXT:      } do {
// CHECK-NEXT:      ^bb0(%[[arg1:.*]]: [[upstreamStateType]], %[[arg2:.*]]: tuple<i32>, %[[arg3:.*]]: tuple<i32>):
// CHECK-NEXT:        %[[V7:.*]] = func.call @sum_tuple(%[[arg2]], %[[arg3]]) : (tuple<i32>, tuple<i32>) -> tuple<i32>
// CHECK-NEXT:        scf.yield %[[arg1]], %[[V7]] : [[upstreamStateType]], tuple<i32>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[V5]]#0, %[[V5]]#1 : [[upstreamStateType]], tuple<i32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[V8:.*]] = arith.constant true
// CHECK-NEXT:    %[[V9:.*]] = arith.xori %[[V8]], %[[V3]] : i1
// CHECK-NEXT:    %[[Va:.*]] = iterators.insertvalue %[[V4]]#0 into %[[ARG0]][0] : !iterators.state<[[upstreamStateType]], tuple<i32>, i1>
// CHECK-NEXT:    %[[Vb:.*]] = iterators.insertvalue %[[V8]] into %[[Va]][2] : !iterators.state<[[upstreamStateType]], tuple<i32>, i1>
// CHECK-NEXT:    return %[[Vb]], %[[V9]], %[[V4]]#1 : !iterators.state<[[upstreamStateType]], tuple<i32>, i1>, i1, tuple<i32>

// CHECK-LABEL: func.func private @iterators.accumulate.open.{{[0-9]+}}(
// CHECK-SAME:        %[[ARG0:.*]]: !iterators.state<[[upstreamStateType:!iterators.state<[^>]*>]], tuple<i32>, i1>) ->
// CHECK-SAME:            !iterators.state<[[upstreamStateType]], tuple<i32>, i1> {
// CHECK-NEXT:    %[[V1:.*]] = iterators.extractvalue %[[ARG0]][0] : !iterators.state<[[upstreamStateType]], tuple<i32>, i1>
// CHECK-NEXT:    %[[V2:.*]] = call @iterators.{{.*}}.open.{{.*}}(%[[V1]]) : ([[upstreamStateType]]) -> [[upstreamStateType]]
// CHECK-NEXT:    %[[V3:.*]] = iterators.insertvalue %[[V2]] into %[[ARG0]][0] : !iterators.state<[[upstreamStateType]], tuple<i32>, i1>
// CHECK-NEXT:    %[[V4:.*]] = arith.constant false
// CHECK-NEXT:    %[[V5:.*]] = iterators.insertvalue %[[V4]] into %[[V3]][2] : !iterators.state<[[upstreamStateType]], tuple<i32>, i1>
// CHECK-NEXT:    return %[[V5]] : !iterators.state<[[upstreamStateType]], tuple<i32>, i1>

// CHECK-LABEL: func.func @main()
func.func @main() {
  // CHECK-DAG:    %[[V0:.*]] = iterators.createstate{{.*}} : [[upstreamStateType:!iterators.state<[^>]*>]]
  %input = "iterators.constantstream"() { value = [] } : () -> (!iterators.stream<tuple<i32>>)

  // CHECK-DAG:    %[[V1:.*]] = tuple.from_elements %{{.*}} : tuple<i32>
  %hundred = arith.constant 0 : i32
  %init_value = tuple.from_elements %hundred : tuple<i32>

  // CHECK-DAG:    %[[V2:.*]] = arith.constant false
  // CHECK-NEXT:   %[[V3:.*]] = iterators.createstate(%[[V0]], %[[V1]], %[[V2]]) : !iterators.state<[[upstreamStateType]], tuple<i32>, i1>
  %accumulated = iterators.accumulate(%input, %init_value) with @sum_tuple
    : (!iterators.stream<tuple<i32>>) -> !iterators.stream<tuple<i32>>
  return
  // CHECK-NEXT:   return
}
