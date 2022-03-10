// RUN: mlir-opt %s -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libruntime_utils%shlibext \
// RUN: | FileCheck %s

func private @iteratorsMakeHeap() -> (!llvm.ptr<i8>)
func private @iteratorsDestroyHeap(!llvm.ptr<i8>) -> ()
func private @iteratorsMakeSampleInputOperator(!llvm.ptr<i8>) -> !llvm.ptr<i8>
func private @iteratorsMakeReduceOperator(!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.ptr<i8>
func private @iteratorsComsumeAndPrint(!llvm.ptr<i8>) -> ()

func @main() {
  %heap = call @iteratorsMakeHeap() : () -> (!llvm.ptr<i8>)
  %input = call @iteratorsMakeSampleInputOperator(%heap) : (!llvm.ptr<i8>) -> (!llvm.ptr<i8>)
  %reduce = call @iteratorsMakeReduceOperator(%heap, %input) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> (!llvm.ptr<i8>)
  // CHECK: (6)
  call @iteratorsComsumeAndPrint(%reduce) : (!llvm.ptr<i8>) -> ()
  call @iteratorsDestroyHeap(%heap) : (!llvm.ptr<i8>) -> ()
  return
}
