// RUN: mlir-proto-opt %s -convert-iterators-to-llvm \
// RUN: | FileCheck --enable-var-scope %s

!element_type = type !llvm.struct<(i32)>

// CHECK-LABEL: func private @iterators.sampleInput.close.{{[0-9]+}}(%{{.*}}: !llvm.struct<"iterators.sample_input_state{{.*}}", (i32)>) -> !llvm.struct<"iterators.sample_input_state{{.*}}", (i32)>
// CHECK-NEXT:    return %[[arg0:.*]] : !llvm.struct<"[[inputStateType:iterators\.sample_input_state.*]]", (i32)>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @iterators.sampleInput.next.{{[0-9]+}}(%{{.*}}: !llvm.struct<"iterators.sample_input_state{{.*}}", (i32)>) -> (!llvm.struct<"iterators.sample_input_state{{.*}}", (i32)>, i1, !llvm.struct<(i32)>)
// CHECK-NEXT:    %[[V0:.*]] = llvm.extractvalue %[[arg0:.*]][0 : index] : !llvm.struct<"[[inputStateType:iterators\.sample_input_state.*]]", (i32)>
// CHECK-NEXT:    %[[V1:.*]] = arith.constant 4 : i32
// CHECK-NEXT:    %[[V2:.*]] = arith.cmpi slt, %[[V0]], %[[V1]] : i32
// CHECK-NEXT:    %[[V3:.*]] = scf.if %[[V2]] -> (!llvm.struct<"[[inputStateType]]", (i32)>) {
// CHECK-NEXT:      %[[V4:.*]] = arith.constant 1 : i32
// CHECK-NEXT:      %[[V5:.*]] = arith.addi %[[V0]], %[[V4]] : i32
// CHECK-NEXT:      %[[V6:.*]] = llvm.insertvalue %[[V5]], %[[arg0]][0 : index] : !llvm.struct<"[[inputStateType]]", (i32)>
// CHECK-NEXT:      scf.yield %[[V6]] : !llvm.struct<"[[inputStateType]]", (i32)>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %[[arg0]] : !llvm.struct<"[[inputStateType]]", (i32)>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[V7:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
// CHECK-NEXT:    %[[V8:.*]] = llvm.insertvalue %[[V0]], %[[V7]][0 : index] : !llvm.struct<(i32)>
// CHECK-NEXT:    return %[[V3]], %[[V2]], %[[V8]] : !llvm.struct<"[[inputStateType]]", (i32)>, i1, !llvm.struct<(i32)>
// CHECK-NEXT:  }

// CHECK-LABEL: func private @iterators.sampleInput.open.{{[0-9]+}}(%{{.*}}: !llvm.struct<"iterators.sample_input_state{{.*}}", (i32)>) -> !llvm.struct<"iterators.sample_input_state{{.*}}", (i32)>
// CHECK-NEXT:    %[[V0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V1:.*]] = llvm.insertvalue %[[V0]], %[[arg0:.*]][0 : index] : !llvm.struct<"[[inputStateType:iterators\.sample_input_state.*]]", (i32)>
// CHECK-NEXT:    return %[[V1]] : !llvm.struct<"[[inputStateType]]", (i32)>
// CHECK-NEXT:  }

func @main() {
  // CHECK-LABEL: func @main()
  %input = "iterators.sampleInput"() : () -> (!iterators.stream<!element_type>)
  // CHECK-NEXT:   %[[V0:.*]] = llvm.mlir.undef : !llvm.struct<"[[inputStateType:iterators\.sample_input_state.*]]", (i32)>
  return
  // CHECK-NEXT:   return
}
// CHECK-NEXT:   }
// CHECK-NEXT: }
