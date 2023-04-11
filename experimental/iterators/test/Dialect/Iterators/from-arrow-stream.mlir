// RUN: iterators-opt %s \
// RUN: | FileCheck %s

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

// CHECK-LABEL: func.func @main(
// CHECK-SAME:                  %[[ARG0:.*]]: !llvm.ptr<[[STREAMTYPE:.*]]>) ->
// CHECK-SAME:                      !iterators.stream<!tabular.tabular_view<i32>> {
// CHECK-NEXT:    %[[V0:fromarrowstream.*]] = iterators.from_arrow_array_stream %[[ARG0]] to !iterators.stream<!tabular.tabular_view<i32>>
// CHECK-NEXT:    return %[[V0]] : !iterators.stream<!tabular.tabular_view<i32>>
func.func @main(%arrow_stream: !llvm.ptr<!arrow_array_stream>) -> !iterators.stream<!tabular.tabular_view<i32>> {
  %tabular_view_stream = iterators.from_arrow_array_stream %arrow_stream to !iterators.stream<!tabular.tabular_view<i32>>
  return %tabular_view_stream : !iterators.stream<!tabular.tabular_view<i32>>
}
