// RUN: mlir-proto-opt %s -convert-iterators-to-llvm \
// RUN: | FileCheck %s

// CHECK:      module {
// CHECK-NEXT:   func private @iteratorsComsumeAndPrint(!llvm.ptr<i8>)
// CHECK-NEXT:   func private @iteratorsDestroyReduceOperator(!llvm.ptr<i8>)
// CHECK-NEXT:   func private @iteratorsMakeReduceOperator(!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK-NEXT:   func private @iteratorsDestroySampleInputOperator(!llvm.ptr<i8>)
// CHECK-NEXT:   func private @iteratorsMakeSampleInputOperator() -> !llvm.ptr<i8>
func @main() {
// CHECK-NEXT:   func @main() {
  %input = "iterators.sampleInput"() : () -> (!iterators.stream<tuple<i32>>)
// CHECK-NEXT:     %[[V0:.*]] = call @iteratorsMakeSampleInputOperator() : () -> !llvm.ptr<i8>
  %reduce = "iterators.reduce"(%input) : (!iterators.stream<tuple<i32>>) -> (!iterators.stream<tuple<i32>>)
// CHECK-NEXT:     %[[V1:.*]] = call @iteratorsMakeReduceOperator(%[[V0]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
  "iterators.sink"(%reduce) : (!iterators.stream<tuple<i32>>) -> ()
// CHECK-NEXT:     call @iteratorsComsumeAndPrint(%[[V1]]) : (!llvm.ptr<i8>) -> ()
  return
// CHECK-NEXT:     call @iteratorsDestroySampleInputOperator(%[[V0]]) : (!llvm.ptr<i8>) -> ()
// CHECK-NEXT:     call @iteratorsDestroyReduceOperator(%[[V1]]) : (!llvm.ptr<i8>) -> ()
// CHECK-NEXT:     return
}
// CHECK-NEXT:   }
// CHECK-NEXT: }
