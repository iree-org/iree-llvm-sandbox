// RUN: iterators-opt %s -convert-iterators-to-llvm \
// RUN: | FileCheck --enable-var-scope %s
!arrow_schema = !llvm.struct<"ArrowSchema", (
  ptr<i8>,  // format
  ptr<i8>,  // name
  ptr<i8>,  // metadata
  i64,      // flags
  i64,      // n_children
  ptr<ptr<struct<"ArrowSchema">>>,              // children
  ptr<struct<"ArrowSchema">>,                   // dictionary
  ptr<func<void (ptr<struct<"ArrowSchema">>)>>, // release
  ptr       // private_data
  )>
!arrow_array = !llvm.struct<"ArrowArray", (
  i64,        // length
  i64,        // null_count
  i64,        // offset
  i64,        // n_buffers
  i64,        // n_children
  ptr<ptr>,   // buffers
  ptr<ptr<struct<"ArrowArray">>>,               // children
  ptr<struct<"ArrowArray">>,                    // dictionary
  ptr<func<void (ptr<struct<"ArrowArray">>)>>,  // release
  ptr         // private_data
  )>
!arrow_array_stream = !llvm.struct<"ArrowArrayStream", (
  ptr<func<i32 (ptr<struct<"ArrowArrayStream">>, ptr<!arrow_schema>)>>, // get_schema
  ptr<func<i32 (ptr<struct<"ArrowArrayStream">>, ptr<!arrow_array>)>>,  // get_next
  ptr<func<ptr<i8> (ptr<struct<"ArrowArrayStream">>)>>, // get_last_error
  ptr<func<void (ptr<struct<"ArrowArrayStream">>)>>,    // release
  ptr   // private_data
  )>

// CHECK-LABEL: func.func private @iterators.from_arrow_array_stream.close.{{[0-9]+}}(
// CHECK-SAME:        %[[ARG0:.*]]: !iterators.state<!llvm.ptr<[[STREAMTYPE:.*]]>, !llvm.ptr<[[SCHEMATYPE:.*]]>, !llvm.ptr<[[ARRAYTYPE:.*]]>>) ->
// CHECK-SAME:            !iterators.state<!llvm.ptr<[[STREAMTYPE]]>, !llvm.ptr<[[SCHEMATYPE]]>, !llvm.ptr<[[ARRAYTYPE]]>> {
// CHECK-NEXT:    %[[V0:.*]] = iterators.extractvalue %[[ARG0]][0] : !iterators.state<!llvm.ptr<[[STREAMTYPE]]>, !llvm.ptr<[[SCHEMATYPE]]>, !llvm.ptr<[[ARRAYTYPE]]>>
// CHECK-NEXT:    %[[V1:.*]] = iterators.extractvalue %[[ARG0]][1] : !iterators.state<!llvm.ptr<[[STREAMTYPE]]>, !llvm.ptr<[[SCHEMATYPE]]>, !llvm.ptr<[[ARRAYTYPE]]>>
// CHECK-NEXT:    %[[V2:.*]] = iterators.extractvalue %[[ARG0]][2] : !iterators.state<!llvm.ptr<[[STREAMTYPE]]>, !llvm.ptr<[[SCHEMATYPE]]>, !llvm.ptr<[[ARRAYTYPE]]>>
// CHECK-NEXT:    llvm.call @mlirIteratorsArrowArrayStreamRelease(%[[V0]]) : (!llvm.ptr<[[STREAMTYPE]]>) -> ()
// CHECK-NEXT:    llvm.call @mlirIteratorsArrowSchemaRelease(%[[V1]]) : (!llvm.ptr<[[SCHEMATYPE]]>) -> ()
// CHECK-NEXT:    llvm.call @mlirIteratorsArrowArrayRelease(%[[V2]]) : (!llvm.ptr<[[ARRAYTYPE]]>) -> ()
// CHECK-NEXT:    return %[[ARG0]] : !iterators.state<!llvm.ptr<[[STREAMTYPE]]>, !llvm.ptr<[[SCHEMATYPE]]>, !llvm.ptr<[[ARRAYTYPE]]>>

// CHECK-LABEL: func.func private @iterators.from_arrow_array_stream.next.{{[0-9]+}}(
// CHECK-SAME:        %[[ARG0:.*]]: !iterators.state<!llvm.ptr<[[STREAMTYPE:.*]]>, !llvm.ptr<[[SCHEMATYPE:.*]]>, !llvm.ptr<[[ARRAYTYPE:.*]]>>) ->
// CHECK-SAME:            (!iterators.state<!llvm.ptr<[[STREAMTYPE]]>, !llvm.ptr<[[SCHEMATYPE]]>, !llvm.ptr<[[ARRAYTYPE]]>>, i1, !llvm.struct<(i64, ptr)>) {
// CHECK-NEXT:    %[[V0:.*]] = iterators.extractvalue %[[ARG0]][0] : !iterators.state<!llvm.ptr<[[STREAMTYPE]]>, !llvm.ptr<[[SCHEMATYPE]]>, !llvm.ptr<[[ARRAYTYPE]]>>
// CHECK-NEXT:    %[[V1:.*]] = iterators.extractvalue %[[ARG0]][1] : !iterators.state<!llvm.ptr<[[STREAMTYPE]]>, !llvm.ptr<[[SCHEMATYPE]]>, !llvm.ptr<[[ARRAYTYPE]]>>
// CHECK-NEXT:    %[[V2:.*]] = iterators.extractvalue %[[ARG0]][2] : !iterators.state<!llvm.ptr<[[STREAMTYPE]]>, !llvm.ptr<[[SCHEMATYPE]]>, !llvm.ptr<[[ARRAYTYPE]]>>
// CHECK-NEXT:    llvm.call @mlirIteratorsArrowArrayRelease(%[[V2]]) : (!llvm.ptr<[[ARRAYTYPE]]>) -> ()
// CHECK-NEXT:    %[[V3:.*]] = llvm.call @mlirIteratorsArrowArrayStreamGetNext(%[[V0]], %[[V2]]) : (!llvm.ptr<[[STREAMTYPE]]>, !llvm.ptr<[[ARRAYTYPE]]>) -> i1
// CHECK-NEXT:    %[[V4:.*]] = arith.constant 0 : i64
// CHECK-NEXT:    %[[V5:.*]] = scf.if %[[V3]] -> (i64) {
// CHECK-NEXT:      %[[V6:.*]] = llvm.call @mlirIteratorsArrowArrayGetSize(%[[V2]]) : (!llvm.ptr<[[ARRAYTYPE]]>) -> i64
// CHECK-NEXT:      scf.yield %[[V6]] : i64
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %[[V4]] : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[V7:.*]]:2 = scf.if %[[V3]] -> (!llvm.ptr, i64) {
// CHECK-NEXT:      %[[V8:.*]] = arith.constant 0 : i64
// CHECK-NEXT:      %[[V9:.*]] = llvm.call @mlirIteratorsArrowArrayGetInt32Column(%[[V2]], %[[V1]], %[[V8]]) : (!llvm.ptr<[[ARRAYTYPE]]>, !llvm.ptr<[[SCHEMATYPE]]>, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:      %[[Va:.*]] = llvm.bitcast %[[V9]] : !llvm.ptr<i32> to !llvm.ptr
// CHECK-NEXT:      scf.yield %[[Va]], %[[V5]] : !llvm.ptr, i64
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[Va:.*]] = llvm.mlir.null : !llvm.ptr
// CHECK-NEXT:      scf.yield %[[Va]], %[[V4]] : !llvm.ptr, i64
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[Vb:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[Vc:.*]] = llvm.insertvalue %[[V7]]#0, %[[Vb]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[Vd:.*]] = llvm.insertvalue %[[V7]]#0, %[[Vc]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[Ve:.*]] = llvm.insertvalue %[[V4]], %[[Vd]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[Vf:.*]] = llvm.insertvalue %[[V7]]#1, %[[Ve]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[Vg:.*]] = llvm.insertvalue %[[V7]]#1, %[[Vf]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[Vh:.*]] = builtin.unrealized_conversion_cast %[[Vg]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xi32>
// CHECK-NEXT:    %[[Vi:.*]] = tabular.view_as_tabular %[[Vh]] : (memref<?xi32>) -> !tabular.tabular_view<i32>
// CHECK-NEXT:    %[[Vj:.*]] = builtin.unrealized_conversion_cast %[[Vi]] : !tabular.tabular_view<i32> to !llvm.struct<(i64, ptr)>
// CHECK-NEXT:    return %[[ARG0]], %[[V3]], %[[Vj]] : !iterators.state<!llvm.ptr<[[STREAMTYPE]]>, !llvm.ptr<[[SCHEMATYPE]]>, !llvm.ptr<[[ARRAYTYPE]]>>, i1, !llvm.struct<(i64, ptr)>

// CHECK-LABEL: func.func private @iterators.from_arrow_array_stream.open.{{[0-9]+}}(
// CHECK-SAME:        %[[ARG0:.*]]: !iterators.state<!llvm.ptr<[[STREAMTYPE:.*]]>, !llvm.ptr<[[SCHEMATYPE:.*]]>, !llvm.ptr<[[ARRAYTYPE:.*]]>>) ->
// CHECK-SAME:            !iterators.state<!llvm.ptr<[[STREAMTYPE]]>, !llvm.ptr<[[SCHEMATYPE]]>, !llvm.ptr<[[ARRAYTYPE]]>> {
// CHECK-NEXT:    %[[V0:.*]] = iterators.extractvalue %[[ARG0]][0] : !iterators.state<!llvm.ptr<[[STREAMTYPE]]>, !llvm.ptr<[[SCHEMATYPE]]>, !llvm.ptr<[[ARRAYTYPE]]>>
// CHECK-NEXT:    %[[V1:.*]] = iterators.extractvalue %[[ARG0]][1] : !iterators.state<!llvm.ptr<[[STREAMTYPE]]>, !llvm.ptr<[[SCHEMATYPE]]>, !llvm.ptr<[[ARRAYTYPE]]>>
// CHECK-NEXT:    llvm.call @mlirIteratorsArrowArrayStreamGetSchema(%[[V0]], %[[V1]]) : (!llvm.ptr<[[STREAMTYPE]]>, !llvm.ptr<[[SCHEMATYPE]]>) -> i32
// CHECK-NEXT:    return %[[ARG0]] : !iterators.state<!llvm.ptr<[[STREAMTYPE]]>, !llvm.ptr<[[SCHEMATYPE]]>, !llvm.ptr<[[ARRAYTYPE]]>>

// CHECK-LABEL: func.func @main(
// CHECK-SAME:                  %[[ARG0:.*]]: !llvm.ptr<[[STREAMTYPE:.*]]>)
// CHECK-NEXT:    %[[V0:.*]] = arith.constant 1 : i64
// CHECK-NEXT:    %[[V1:.*]] = llvm.alloca %[[V0]] x !llvm.[[ARRAYTYPE:.*]] : (i64) ->
// CHECK-SAME:                    !llvm.ptr<[[ARRAYTYPE]]>
// CHECK-NEXT:    %[[V2:.*]] = llvm.alloca %[[V0]] x !llvm.[[SCHEMATYPE:.*]] : (i64) ->
// CHECK-SAME:                    !llvm.ptr<[[SCHEMATYPE]]>
// CHECK-NEXT:    %[[V3:.*]] = arith.constant 0 : i8
// CHECK-NEXT:    %[[V4:.*]] = arith.constant false
// CHECK-NEXT:    %[[V5:.*]] = arith.constant 80 : i64
// CHECK-NEXT:    %[[V6:.*]] = arith.constant 72 : i64
// CHECK-NEXT:    "llvm.intr.memset"(%[[V1]], %[[V3]], %[[V5]], %[[V4]]) : (!llvm.ptr<[[ARRAYTYPE]]>, i8, i64, i1) -> ()
// CHECK-NEXT:    "llvm.intr.memset"(%[[V2]], %[[V3]], %[[V6]], %[[V4]]) : (!llvm.ptr<[[SCHEMATYPE]]>, i8, i64, i1) -> ()
// CHECK-NEXT:    %[[V7:.*]] = iterators.createstate(%[[ARG0]], %[[V2]], %[[V1]]) : !iterators.state<!llvm.ptr<[[STREAMTYPE]]>, !llvm.ptr<[[SCHEMATYPE]]>, !llvm.ptr<[[ARRAYTYPE]]>>
// CHECK-NEXT:    return
func.func @main(%arrow_stream: !llvm.ptr<!arrow_array_stream>) {
  %tabular_view_stream = iterators.from_arrow_array_stream %arrow_stream to !iterators.stream<!tabular.tabular_view<i32>>
  return
}
