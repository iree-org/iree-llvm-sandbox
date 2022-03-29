// RUN: mlir-proto-opt %s -convert-iterators-to-std \
// RUN: | FileCheck %s

func @main() {
  %input = "iterators.sampleInput"() : () -> (!iterators.iterator<tuple<i32>>)
  %reduce = "iterators.reduce"(%input) : (!iterators.iterator<tuple<i32>>) -> (!iterators.iterator<tuple<i32>>)
  "iterators.sink"(%reduce) : (!iterators.iterator<tuple<i32>>) -> ()
  return
}
// CHECK:      module {
// CHECK-NEXT:   func private @iteratorsMakeSampleInputOperator() -> !llvm.ptr<i8>
// CHECK-NEXT:   func private @iteratorsDestroySampleInputOperator(!llvm.ptr<i8>)
// CHECK-NEXT:   func private @iteratorsMakeReduceOperator(!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK-NEXT:   func private @iteratorsDestroyReduceOperator(!llvm.ptr<i8>)
// CHECK-NEXT:   func private @iteratorsComsumeAndPrint(!llvm.ptr<i8>)
// CHECK-NEXT:   func @main() {
// CHECK-NEXT:     %0 = call @iteratorsMakeSampleInputOperator() : () -> !llvm.ptr<i8>
// CHECK-NEXT:     %1 = call @iteratorsMakeReduceOperator(%0) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK-NEXT:     call @iteratorsComsumeAndPrint(%1) : (!llvm.ptr<i8>) -> ()
// CHECK-NEXT:     call @iteratorsDestroySampleInputOperator(%0) : (!llvm.ptr<i8>) -> ()
// CHECK-NEXT:     call @iteratorsDestroyReduceOperator(%1) : (!llvm.ptr<i8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }
