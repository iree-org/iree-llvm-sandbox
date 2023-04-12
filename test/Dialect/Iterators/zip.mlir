// RUN: iterators-opt %s \
// RUN: | FileCheck %s

func.func @main(%stream_i32 : !iterators.stream<i32>,
                %stream_i64 : !iterators.stream<i64>) {
  // CHECK-LABEL: func.func @main(
  // CHECK-SAME:    %[[arg0:.*]]: !iterators.stream<i32>, %[[arg1:.*]]: !iterators.stream<i64>) {
  %zipped = iterators.zip %stream_i32, %stream_i64 :
              (!iterators.stream<i32>, !iterators.stream<i64>)
                -> !iterators.stream<tuple<i32, i64>>
  // CHECK-NEXT:    %[[V0:zipped.*]] = iterators.zip %[[arg0]], %[[arg1]] : (!iterators.stream<i32>, !iterators.stream<i64>) -> !iterators.stream<tuple<i32, i64>>
  return
  // CHECK-NEXT:    return
}
// CHECK-NEXT:    }
