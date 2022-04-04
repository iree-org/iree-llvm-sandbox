// RUN: mlir-proto-opt %s -convert-iterators-to-std \
// RUN: | FileCheck %s

func @main() {
  %emtpyOptional = "iterators.emptyoptional"() : () -> !iterators.optional<i32>
  // CHECK:      %[[V0:.*]] = llvm.mlir.undef : !llvm.struct<"Optional[[S0:.*]]", (i1, i32)>
  // CHECK-NEXT: %[[V1:.*]] = arith.constant false
  // CHECK-NEXT: %[[V2:.*]] = llvm.insertvalue %[[V1]], %[[V0]][0 : index] : !llvm.struct<"Optional[[S0]]", (i1, i32)>

  %fourtyTwo = arith.constant 42 : i32
  // CHECK-NEXT: %[[V3:.*]] = arith.constant 42 : i32
  %optionalInt = "iterators.insertvalue"(%emtpyOptional, %fourtyTwo) : (!iterators.optional<i32>, i32) -> !iterators.optional<i32>
  // CHECK-NEXT: %[[V4:.*]] = llvm.insertvalue %[[V3]], %[[V2]][1 : index] : !llvm.struct<"Optional[[S0]]", (i1, i32)>

  %hasValue = "iterators.hasvalue"(%optionalInt) : (!iterators.optional<i32>) -> i1
  // CHECK-NEXT: %[[V5:.*]] = llvm.extractvalue %[[V4]][0 : index] : !llvm.struct<"Optional[[S0]]", (i1, i32)>

  %value = "iterators.extractvalue"(%optionalInt) : (!iterators.optional<i32>) -> i32
  // CHECK-NEXT: %[[V6:.*]] = llvm.extractvalue %[[V4]][1 : index] : !llvm.struct<"Optional[[S0]]", (i1, i32)>
  return
}
