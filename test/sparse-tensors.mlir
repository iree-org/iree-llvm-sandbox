// RUN: iterators-opt %s \
// RUN:   -pre-sparsification-rewrite \
// RUN:   -sparsification \
// RUN:   -sparse-tensor-conversion \
// RUN:   -canonicalize -cse \
// RUN:   -one-shot-bufferize \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -expand-strided-metadata \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-cf-to-llvm \
// RUN:   -reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext \
// RUN: | FileCheck %s

#SV = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>
func.func @main() {
  %dt = arith.constant dense<[0, 1, 0, 0, 2, 0, 0, 3]> : tensor<8xi64>
  %sp = sparse_tensor.convert %dt : tensor<8xi64> to tensor<8xi64, #SV>
  %coordinates = sparse_tensor.coordinates %sp { level = 0 : index }
      : tensor<8xi64, #SV> to memref<?xindex>
  %positions = sparse_tensor.positions %sp { level = 0 : index }
      : tensor<8xi64, #SV> to memref<?xindex>
  %values = sparse_tensor.values %sp
      : tensor<8xi64, #SV> to memref<?xi64>
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %num_coordinates_idx = memref.dim %coordinates, %zero : memref<?xindex>
  %num_positions_idx = memref.dim %positions, %zero : memref<?xindex>
  %num_values_idx = memref.dim %values, %zero : memref<?xi64>
  %num_coordinates = arith.index_cast %num_coordinates_idx : index to i64
  %num_positions = arith.index_cast %num_positions_idx : index to i64
  %num_values = arith.index_cast %num_values_idx : index to i64

  // CHECK-LABEL: num_coordinates:
  // CHECK-SAME:  3
  // CHECK-NEXT:  1
  // CHECK-NEXT:  4
  // CHECK-NEXT:  7
  iterators.print("num_coordinates: " nonl)
  iterators.print(%num_coordinates) : i64
  scf.for %i = %zero to %num_coordinates_idx step %one {
    %0 = memref.load %coordinates[%i] : memref<?xindex>
    %1 = arith.index_cast %0 : index to i64
    iterators.print("  " nonl)
    iterators.print(%1) : i64
  }

  // CHECK-LABEL: num_positions:
  // CHECK-SAME:  2
  // CHECK-NEXT:  0
  // CHECK-NEXT:  3
  iterators.print("num_positions: " nonl)
  iterators.print(%num_positions) : i64
  scf.for %i = %zero to %num_positions_idx step %one {
    %0 = memref.load %positions[%i] : memref<?xindex>
    %1 = arith.index_cast %0 : index to i64
    iterators.print("  " nonl)
    iterators.print(%1) : i64
  }

  // CHECK-LABEL: num_values:
  // CHECK-SAME:  3
  // CHECK-NEXT:  1
  // CHECK-NEXT:  2
  // CHECK-NEXT:  3
  iterators.print("num_values: " nonl)
  iterators.print(%num_values) : i64
  scf.for %i = %zero to %num_values_idx step %one {
    %0 = memref.load %values[%i] : memref<?xi64>
    iterators.print("  " nonl)
    iterators.print(%0) : i64
  }

  return
}
