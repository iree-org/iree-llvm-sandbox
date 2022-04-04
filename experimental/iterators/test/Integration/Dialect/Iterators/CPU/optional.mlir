// RUN: mlir-proto-opt -convert-iterators-to-std \
// RUN:                -convert-arith-to-llvm -convert-func-to-llvm %s \
// RUN: | mlir-cpu-runner -e main -entry-point-result=i32 \
// RUN: | FileCheck %s

func @main() -> i32 {
  %emtpyOptional = "iterators.emptyoptional"() : () -> !iterators.optional<i32>
  %fourtyTwo = arith.constant 42 : i32
  %optionalInt = "iterators.insertvalue"(%emtpyOptional, %fourtyTwo) : (!iterators.optional<i32>, i32) -> !iterators.optional<i32>
  %hasValue = "iterators.hasvalue"(%optionalInt) : (!iterators.optional<i32>) -> i1
  %value = "iterators.extractvalue"(%optionalInt) : (!iterators.optional<i32>) -> i32
  // CHECK: 42
  return %value : i32
}
