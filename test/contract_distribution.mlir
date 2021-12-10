// RUN: mlir-proto-opt %s -vector-propagate-distribution  |\
// RUN: mlir-opt -canonicalize  |\
// RUN: mlir-opt -test-vector-multi-reduction-lowering-patterns -convert-linalg-to-loops -convert-vector-to-scf -lower-affine -convert-scf-to-std \
// RUN: -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm -canonicalize \
// RUN: -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void  \
// RUN:   -shared-libs=%iree_runners_test_dir/libruntime-support%shlibext | \
// RUN: tee | FileCheck %s

// RUN: mlir-proto-opt %s -vector-propagate-distribution |\
// RUN: mlir-opt -canonicalize -cse | FileCheck %s -check-prefix=TRANSFORM

func private @print_memref_f32(memref<*xf32>)

func @alloc_2d_filled_f32(%s1 : index, %s2 : index, %f : f32) -> memref<?x?xf32> {
  %buf = memref.alloc(%s1, %s2) : memref<?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %arg2 = %c0 to %s1 step %c1 {
   scf.for %arg3 = %c0 to %s2 step %c1 {
      %tmp = arith.index_cast %arg2 : index to i32
      %tmp1 = arith.sitofp %tmp : i32 to f32
      %tmp2 = arith.addf %tmp1, %f : f32
      memref.store %tmp2, %buf[%arg2, %arg3] : memref<?x?xf32>
    }
  }
  return %buf : memref<?x?xf32>
}

#matmat_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
#matmat_trait = {
  indexing_maps = #matmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// TRANSFORM: vector.transfer_read {{.*}} : memref<?x?xf32>, vector<4x4xf32>
// TRANSFORM: scf.for {{.*}} -> (vector<4x4xf32>) {
// TRANSFORM:   vector.transfer_read {{.*}} : memref<?x?xf32>, vector<4x8xf32>
// TRANSFORM:   vector.transfer_read {{.*}} : memref<?x?xf32>, vector<8x4xf32>
// TRANSFORM:   vector.contract {{.*}} : vector<4x8xf32>, vector<8x4xf32> into vector<4x4xf32>
// TRANSFORM:   scf.yield %{{.*}} : vector<4x4xf32>
// TRANSFORM: }
// TRANSFORM: vector.transfer_write {{.*}} : vector<4x4xf32>, memref<?x?xf32>
func @simt_func(%idx : index, %idy : index, %A : memref<?x?xf32>,
                %B : memref<?x?xf32>, %C : memref<?x?xf32>) {
  %c128 = arith.constant 128 : index
  %c384 = arith.constant 384 : index
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %7 = vector.transfer_read %C[%c0, %c0], %cst {in_bounds = [true, true]}
                  : memref<?x?xf32>, vector<128x128xf32>
  %8 = scf.for %arg0 = %c0 to %c128 step %c8 iter_args(%arg1 = %7) -> (vector<128x128xf32>) {
    %11 = vector.transfer_read %A[%c0, %arg0], %cst {in_bounds = [true, true]} : memref<?x?xf32>, vector<128x8xf32>
    %12 = vector.transfer_read %B[%arg0, %c0], %cst {in_bounds = [true, true]} : memref<?x?xf32>, vector<8x128xf32>
    %13 = vector.contract #matmat_trait %11, %12, %arg1 : vector<128x8xf32>, vector<8x128xf32> into vector<128x128xf32>
    scf.yield %13 : vector<128x128xf32>
  }
  %ext = vector.extract_map %8[%idx, %idy] : vector<128x128xf32> to vector<4x4xf32>
  %ins = vector.insert_map %ext, %8[%idx, %idy] : vector<4x4xf32> into vector<128x128xf32>
  vector.transfer_write %ins, %C[%c0, %c0] {in_bounds = [true, true]} : vector<128x128xf32>, memref<?x?xf32>
  return
}

// Large vector addf that can be broken down into a loop of smaller vector addf.
func @main() {
  %cf0 = arith.constant 0.0 : f32
  %cf1 = arith.constant 1.0 : f32
  %cf2 = arith.constant 2.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %out = memref.alloc(%c128, %c128) : memref<?x?xf32>
  linalg.fill(%cf0, %out) : f32, memref<?x?xf32>
  %in1 = call @alloc_2d_filled_f32(%c128, %c128, %cf1) : (index, index, f32) -> memref<?x?xf32>
  %in2 = call @alloc_2d_filled_f32(%c128, %c128, %cf0) : (index, index, f32) -> memref<?x?xf32>
  scf.for %i = %c0 to %c32 step %c1 {
    scf.for %j = %c0 to %c32 step %c1 {
      call @simt_func(%i, %j, %in1, %in2, %out) : (index, index, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    }
  }
  %converted = memref.cast %out : memref<?x?xf32> to memref<*xf32>
  call @print_memref_f32(%converted): (memref<*xf32>) -> ()
  // CHECK:      Unranked{{.*}}data =
  // CHECK-NEXT: [8128,
  memref.dealloc %out : memref<?x?xf32>
  memref.dealloc %in1 : memref<?x?xf32>
  memref.dealloc %in2 : memref<?x?xf32>
  return
}
