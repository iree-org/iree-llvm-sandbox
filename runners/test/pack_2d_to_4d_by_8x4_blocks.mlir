// RUN: mlir-proto-opt %s -linalg-comprehensive-bufferize-inplace -debug |\
// RUN: mlir-opt -convert-vector-to-scf -lower-affine -convert-linalg-to-loops |\
// RUN: mlir-opt -canonicalize -convert-scf-to-std -lower-affine -convert-vector-to-llvm -convert-std-to-llvm -snapshot-op-locations='filename=/tmp/intermediate_llvm.mlir' |\
// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%iree_runners_test_dir/libruntime-support%shlibext |\
// RUN: FileCheck %s

#input_generator_accesses = [
  affine_map<(i, j) -> (i, j)>
]

#input_generator_trait = {
  indexing_maps = #input_generator_accesses,
  iterator_types = ["parallel", "parallel"]
}

#pack_2d_to_4d_by_8x4_blocks_accesses = [
  affine_map<(i, j, u, v) -> (8 * i + u, 4 * j + v)>,
  affine_map<(i, j, u, v) -> (i, j, u, v)>
]

#pack_2d_to_4d_by_8x4_blocks_trait = {
  indexing_maps = #pack_2d_to_4d_by_8x4_blocks_accesses,
  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
}

func @main() {
  %c0 = constant 0 : index
  %v0 = constant 0.0 : f32
  // Generate some input 2d tensor %a. We fill it with values of row-major offsets
  // to make it easy to track how the subsequent packing op shuffled data.
  %dst_init = linalg.init_tensor [32, 32] : tensor<32x32xf32>
  %a = linalg.indexed_generic #input_generator_trait outs(%dst_init : tensor<32x32xf32>) {
    ^bb0(%i : index, %j : index, %x : f32):
      %c32 = constant 32 : index
      %m = std.muli %i, %c32 : index
      %s = std.addi %m, %j : index
      %k = std.index_cast %s : index to i32
      %f = std.sitofp %k : i32 to f32
      linalg.yield %f : f32
  } -> tensor<32x32xf32>
  // Perform the packing of the above 2d tensor %a into a 4d tensor %a4d
  // packed by 8x4 blocks.
  %dst4d_init = linalg.init_tensor [4, 8, 8, 4] : tensor<4x8x8x4xf32>
  %a4d = linalg.generic #pack_2d_to_4d_by_8x4_blocks_trait ins(%a : tensor<32x32xf32>) outs(%dst4d_init : tensor<4x8x8x4xf32>) {
    ^bb0(%x : f32, %y : f32):
      linalg.yield %x : f32
  } -> tensor<4x8x8x4xf32>
  // Read and print out the 4d tensor.
  %val = vector.transfer_read %a4d[%c0, %c0, %c0, %c0], %v0: tensor<4x8x8x4xf32>, vector<4x8x8x4xf32>
  vector.print %val: vector<4x8x8x4xf32>

  return
}

// CHECK: ( (
// CHECK-SAME: ( ( 0, 1, 2, 3 ), ( 32, 33, 34, 35 ), ( 64, 65, 66, 67 ), ( 96, 97, 98, 99 ), ( 128, 129, 130, 131 ), ( 160, 161, 162, 163 ), ( 192, 193, 194, 195 ), ( 224, 225, 226, 227 ) ),
// CHECK-SAME: ( ( 4, 5, 6, 7 ), ( 36, 37, 38, 39 ), ( 68, 69, 70, 71 ), ( 100, 101, 102, 103 ), ( 132, 133, 134, 135 ), ( 164, 165, 166, 167 ), ( 196, 197, 198, 199 ), ( 228, 229, 230, 231 ) ),
// CHECK-SAME: ( ( 8, 9, 10, 11 ), ( 40, 41, 42, 43 ), ( 72, 73, 74, 75 ), ( 104, 105, 106, 107 ), ( 136, 137, 138, 139 ), ( 168, 169, 170, 171 ), ( 200, 201, 202, 203 ), ( 232, 233, 234, 235 ) ),
// CHECK-SAME: ( ( 12, 13, 14, 15 ), ( 44, 45, 46, 47 ), ( 76, 77, 78, 79 ), ( 108, 109, 110, 111 ), ( 140, 141, 142, 143 ), ( 172, 173, 174, 175 ), ( 204, 205, 206, 207 ), ( 236, 237, 238, 239 ) ),
// CHECK-SAME: ( ( 16, 17, 18, 19 ), ( 48, 49, 50, 51 ), ( 80, 81, 82, 83 ), ( 112, 113, 114, 115 ), ( 144, 145, 146, 147 ), ( 176, 177, 178, 179 ), ( 208, 209, 210, 211 ), ( 240, 241, 242, 243 ) ),
// CHECK-SAME: ( ( 20, 21, 22, 23 ), ( 52, 53, 54, 55 ), ( 84, 85, 86, 87 ), ( 116, 117, 118, 119 ), ( 148, 149, 150, 151 ), ( 180, 181, 182, 183 ), ( 212, 213, 214, 215 ), ( 244, 245, 246, 247 ) ),
// CHECK-SAME: ( ( 24, 25, 26, 27 ), ( 56, 57, 58, 59 ), ( 88, 89, 90, 91 ), ( 120, 121, 122, 123 ), ( 152, 153, 154, 155 ), ( 184, 185, 186, 187 ), ( 216, 217, 218, 219 ), ( 248, 249, 250, 251 ) ),
// CHECK-SAME: ( ( 28, 29, 30, 31 ), ( 60, 61, 62, 63 ), ( 92, 93, 94, 95 ), ( 124, 125, 126, 127 ), ( 156, 157, 158, 159 ), ( 188, 189, 190, 191 ), ( 220, 221, 222, 223 ), ( 252, 253, 254, 255 ) ) ),
// CHECK-SAME: (
// CHECK-SAME: ( ( 256, 257, 258, 259 ), ( 288, 289, 290, 291 ), ( 320, 321, 322, 323 ), ( 352, 353, 354, 355 ), ( 384, 385, 386, 387 ), ( 416, 417, 418, 419 ), ( 448, 449, 450, 451 ), ( 480, 481, 482, 483 ) ),
// CHECK-SAME: ( ( 260, 261, 262, 263 ), ( 292, 293, 294, 295 ), ( 324, 325, 326, 327 ), ( 356, 357, 358, 359 ), ( 388, 389, 390, 391 ), ( 420, 421, 422, 423 ), ( 452, 453, 454, 455 ), ( 484, 485, 486, 487 ) ),
// CHECK-SAME: ( ( 264, 265, 266, 267 ), ( 296, 297, 298, 299 ), ( 328, 329, 330, 331 ), ( 360, 361, 362, 363 ), ( 392, 393, 394, 395 ), ( 424, 425, 426, 427 ), ( 456, 457, 458, 459 ), ( 488, 489, 490, 491 ) ),
//
// ...snip...
//
// CHECK-SAME: ( ( 788, 789, 790, 791 ), ( 820, 821, 822, 823 ), ( 852, 853, 854, 855 ), ( 884, 885, 886, 887 ), ( 916, 917, 918, 919 ), ( 948, 949, 950, 951 ), ( 980, 981, 982, 983 ), ( 1012, 1013, 1014, 1015 ) ),
// CHECK-SAME: ( ( 792, 793, 794, 795 ), ( 824, 825, 826, 827 ), ( 856, 857, 858, 859 ), ( 888, 889, 890, 891 ), ( 920, 921, 922, 923 ), ( 952, 953, 954, 955 ), ( 984, 985, 986, 987 ), ( 1016, 1017, 1018, 1019 ) ),
// CHECK-SAME: ( ( 796, 797, 798, 799 ), ( 828, 829, 830, 831 ), ( 860, 861, 862, 863 ), ( 892, 893, 894, 895 ), ( 924, 925, 926, 927 ), ( 956, 957, 958, 959 ), ( 988, 989, 990, 991 ), ( 1020, 1021, 1022, 1023 ) ) ) )
