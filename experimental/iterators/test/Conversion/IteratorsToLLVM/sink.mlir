// RUN: mlir-proto-opt %s -convert-iterators-to-llvm \
// RUN: | FileCheck --enable-var-scope %s

!intTuple = type !llvm.struct<(i32)>

func @main() {
  // CHECK-LABEL: func @main()
  %input = "iterators.sampleInput"() : () -> (!iterators.stream<!intTuple>)
  %reduce = "iterators.reduce"(%input) : (!iterators.stream<!intTuple>) -> (!iterators.stream<!intTuple>)
  "iterators.sink"(%reduce) : (!iterators.stream<!intTuple>) -> ()
  // CHECK:         %[[V1:.*]] = call @[[rootIteratorName:.*]].Open.{{[0-9]+}}(%[[V0:.*]]) : ([[rootStateType:.*]]) -> [[rootStateType]]
  // CHECK-NEXT:    %[[V2:.*]]:3 = scf.while (%[[arg0:.*]] = %[[V1]]) : ([[rootStateType]]) -> ([[rootStateType]], i1, !llvm.struct<(i32)>) {
  // CHECK-NEXT:      %[[V4:.*]]:3 = call @[[rootIteratorName]].Next.{{[0-9]+}}(%[[arg0]]) : ([[rootStateType]]) -> ([[rootStateType]], i1, !llvm.struct<(i32)>)
  // CHECK-NEXT:      scf.condition(%[[V4]]#1) %[[V4]]#0, %[[V4]]#1, %[[V4]]#2 : [[rootStateType]], i1, !llvm.struct<(i32)>
  // CHECK-NEXT:    } do {
  // CHECK-NEXT:    ^[[bb0:.*]](%[[arg1:.*]]: [[rootStateType]], %arg1: i1, %arg2: !llvm.struct<(i32)>):
  // CHECK-NEXT:      %[[V4]] = llvm.mlir.addressof @frmt_spec.anonymous_tuple : !llvm.ptr<array<6 x i8>>
  // CHECK-NEXT:      %[[V5:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT:      %[[V6:.*]] = llvm.getelementptr %[[V4]][%[[V5]], %[[V5]]] : (!llvm.ptr<array<6 x i8>>, i64, i64) -> !llvm.ptr<i8>
  // CHECK-NEXT:      %[[V7:.*]] = llvm.extractvalue %arg2[0 : index] : !llvm.struct<(i32)>
  // CHECK-NEXT:      %[[V8:.*]] = llvm.call @printf(%[[V6]], %[[V7]]) : (!llvm.ptr<i8>, i32) -> i32
  // CHECK-NEXT:      scf.yield %[[arg1]] : [[rootStateType]]
  // CHECK-NEXT:    }
  // CHECK-NEXT:    %[[V3:.*]] = call @[[rootIteratorName]].Close.{{[0-9]+}}(%[[V2]]#0) : ([[rootStateType]]) -> [[rootStateType]]
  return
  // CHECK-NEXT:   return
}
// CHECK-NEXT:   }
// CHECK-NEXT: }
