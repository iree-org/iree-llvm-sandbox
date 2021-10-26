// RUN: mlir-proto-opt %s\
// RUN: -linalg-tensor-codegen-driver="anchor-func=reduce anchor-op=linalg.generic fuse-fill-into-reduction tile-sizes=24,16" \
// RUN: -canonicalize -cse |\
// RUN: FileCheck %s

func @reduce(%input: tensor<2400x1600xf32>, %output: tensor<2400xf32>) -> tensor<2400xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index

  %fill = linalg.fill(%cst, %output) : f32, tensor<2400xf32> -> tensor<2400xf32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%input : tensor<2400x1600xf32>)
    outs(%fill : tensor<2400xf32>) {
  ^bb0(%in: f32, %out: f32):
    %add = arith.addf %in, %out : f32
    linalg.yield %add : f32
  } -> tensor<2400xf32>
  return %sum : tensor<2400xf32>
}

// CHECK:      linalg.fill
// CHECK:      scf.for %[[I:.*]] = %c0 to %c2400 step %c24
// CHECK:        scf.for %[[J:.*]] = %c0 to %c1600 step %c16
// CHECK-NEXT:     tensor.extract_slice %{{.*}}[%[[I]], %[[J]]] [24, 16] [1, 1]
// CHECK-NEXT:     tensor.extract_slice %{{.*}}[%[[I]]] [24] [1]
// CHECK-NEXT:     linalg.init_tensor [24]
// CHECK-NEXT:     linalg.fill
// CHECK-NEXT:     linalg.generic
// CHECK:          linalg.generic
// CHECK:          tensor.insert_slice
