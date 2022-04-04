// Test that we can parse optionals without errors
// RUN: mlir-proto-opt %s

func @main() {
  %emtpyOptional = "iterators.emptyoptional"() : () -> !iterators.optional<i32>
  %fourtyTwo = arith.constant 42 : i32
  %optionalInt = "iterators.insertvalue"(%emtpyOptional, %fourtyTwo) : (!iterators.optional<i32>, i32) -> !iterators.optional<i32>
  %hasValue = "iterators.hasvalue"(%optionalInt) : (!iterators.optional<i32>) -> i1
  %valueOrZero = scf.if %hasValue -> i32 {
    %value = "iterators.extractvalue"(%optionalInt) : (!iterators.optional<i32>) -> i32
    scf.yield %value : i32
  } else {
    %zero = arith.constant 0 : i32
    scf.yield %zero : i32
  }
  return
}
