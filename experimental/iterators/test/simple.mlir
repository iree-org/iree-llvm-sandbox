// RUN: mlir-opt %s -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libruntime_utils%shlibext \
// RUN: | FileCheck %s

func private @oneWay() -> (i64)
func private @otherWay(i64)

func @main() {
  // CHECK: Create a dummy pointer 0xdeadbee
  %p = call @oneWay() : () -> (i64)
  // CHECK: Use a dummy pointer 0xdeadbee
  call @otherWay(%p) : (i64) -> ()
  return
}
