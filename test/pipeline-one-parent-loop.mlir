// RUN: mlir-proto-opt %s -pipeline-one-parent-loop="anchor-func=test anchor-op=scf.yield parent-loop-num=1 II=10 read-latency=20" | \
// RUN: FileCheck %s

// CHECK-LABEL: func @test
func @test(%input: tensor<1000xf32>, %o: tensor<1000xf32>) -> tensor<1000xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 1 : index
  %c250 = arith.constant 250 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant dense<1.000000e+00> : vector<4xf32>

  // CHECK: %[[C248:.*]] = arith.constant 248 : index
  // CHECK: vector.transfer_read {{.*}} : tensor<1000xf32>, vector<4xf32>
  // CHECK: vector.transfer_read {{.*}} : tensor<1000xf32>, vector<4xf32>
  // CHECK: scf.for %{{.*}} to %[[C248]] step %{{.*}} iter_args(%arg3 = %arg1, %arg4 = %0, %arg5 = %1) -> (tensor<1000xf32>, vector<4xf32>, vector<4xf32>) {
  %out = scf.for %i = %c0 to %c250 step %c4 iter_args(%t0 = %o) -> (tensor<1000xf32>) {
    %a = vector.transfer_read %input[%i], %cst_0 : tensor<1000xf32>, vector<4xf32>
    %b = arith.addf %a, %cst_1 : vector<4xf32>
    %t1 = vector.transfer_write %b, %t0[%i] : vector<4xf32>, tensor<1000xf32>
    scf.yield %t1 : tensor<1000xf32>
  }
  
  // CHECK: arith.addf
  // CHECK: vector.transfer_write {{.*}} : vector<4xf32>, tensor<1000xf32>
  // CHECK: arith.addf
  // CHECK: vector.transfer_write {{.*}} : vector<4xf32>, tensor<1000xf32>
  return %out: tensor<1000xf32>
}
