// RUN: iterators-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -inline -decompose-iterator-states -canonicalize \
// RUN:   -one-shot-bufferize=bufferize-function-boundaries \
// RUN:   -expand-strided-metadata -finalize-memref-to-llvm \
// RUN:   -lower-affine -canonicalize \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-func-to-llvm \
// RUN:   -canonicalize \
// RUN:   -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

!struct_i32i32 = !llvm.struct<(i32, i32)>

func.func private @tensor2xi32_to_struct(%input : tensor<2xi32>) -> !struct_i32i32 {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %i0 = tensor.extract %input[%zero] : tensor<2xi32>
  %i1 = tensor.extract %input[%one] : tensor<2xi32>
  %undef = llvm.mlir.undef : !struct_i32i32
  %inserted = llvm.insertvalue %i0, %undef[0] : !struct_i32i32
  %result = llvm.insertvalue %i1, %inserted[1] : !struct_i32i32
  return %result : !struct_i32i32
}

func.func @test_tensor_to_stream_simple_static() {
  iterators.print("test_tensor_to_stream_simple_static")
  %tensor = arith.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
  %stream = iterators.tensor_to_stream %tensor :
                tensor<6xi32> to !iterators.stream<tensor<2xi32>>
  %mapped = "iterators.map"(%stream) {mapFuncRef = @tensor2xi32_to_struct}
    : (!iterators.stream<tensor<2xi32>>) -> (!iterators.stream<!struct_i32i32>)
  "iterators.sink"(%mapped) : (!iterators.stream<!struct_i32i32>) -> ()
  // CHECK-LABEL: test_tensor_to_stream_simple_static
  // CHECK-NEXT:  (1, 2)
  // CHECK-NEXT:  (3, 4)
  // CHECK-NEXT:  (5, 6)
  // CHECK-NEXT:  -
  return
}

func.func @test_tensor_to_stream_simple_dynamic(%tensor : tensor<?xi32>) {
  iterators.print("test_tensor_to_stream_simple_dynamic")
  %stream = iterators.tensor_to_stream %tensor :
                tensor<?xi32> to !iterators.stream<tensor<2xi32>>
  %mapped = "iterators.map"(%stream) {mapFuncRef = @tensor2xi32_to_struct}
    : (!iterators.stream<tensor<2xi32>>) -> (!iterators.stream<!struct_i32i32>)
  "iterators.sink"(%mapped) : (!iterators.stream<!struct_i32i32>) -> ()
  // CHECK-LABEL: test_tensor_to_stream_simple_dynamic
  // CHECK-NEXT:  (11, 12)
  // CHECK-NEXT:  (13, 14)
  // CHECK-NEXT:  (15, 16)
  // CHECK-NEXT:  -
  return
}

func.func @main() {
  func.call @test_tensor_to_stream_simple_static() : () -> ()

  %tensor = arith.constant dense<[11, 12, 13, 14, 15, 16]> : tensor<6xi32>
  %dtensor = tensor.cast %tensor : tensor<6xi32> to tensor<?xi32>
  func.call @test_tensor_to_stream_simple_dynamic(%dtensor) : (tensor<?xi32>) -> ()

  return
}
