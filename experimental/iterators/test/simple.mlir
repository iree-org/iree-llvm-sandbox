// TODO: this currently does not propertly connect to an actual execution

// R-UN: mlir-opt %s -convert-std-to-llvm -reconcile-unrealized-casts | \
// R-UN: mlir-cpu-runner -e main -entry-point-result=void \
// R-UN:   -shared-libs=$(pwd)/lib/Utils/libruntime_utils.so \
// R-UN: | FileCheck %s

func private @oneWay() -> (i64)
func private @otherWay(i64)

func @main() {
  // CHECK: Create a dummy pointer 0xdeadbee
  %p = call @oneWay() : () -> (i64)
  // CHECK: Use a dummy pointer 0xdeadbee
  call @otherWay(%p) : (i64) -> ()
  return
}