// RUN: mlir-proto-opt %s -split-input-file -verify-diagnostics

func @wrong_num_outputs(%laneid: index) {
  // expected-error@+1 {{'vector_ext.warp_execute_on_lane_0' op expected same number of yield operands and return values.}}
  %2 = vector_ext.warp_execute_on_lane_0(%laneid)[64] -> (vector<4xi32>) {
  }
  return
}

// -----

func @wrong_num_inputs(%laneid: index) {
  // expected-error@+1 {{'vector_ext.warp_execute_on_lane_0' op expected same number op arguments and block arguments.}}
  vector_ext.warp_execute_on_lane_0(%laneid)[64] {
  ^bb0(%arg0 : vector<128xi32>) :
  }
  return
}

// -----

func @warp_wrong_return_distribution(%laneid: index) {
  // expected-error@+1 {{'vector_ext.warp_execute_on_lane_0' op incompatible distribution dimensions from 'vector<128xi32>' to 'vector<4xi32>'}}
  %2 = vector_ext.warp_execute_on_lane_0(%laneid)[64] -> (vector<4xi32>) {
    %0 = arith.constant dense<2>: vector<128xi32>
    vector_ext.yield %0 : vector<128xi32>
  }
  return
}


// -----

func @warp_wrong_arg_distribution(%laneid: index, %v0 : vector<4xi32>) {
  // expected-error@+1 {{'vector_ext.warp_execute_on_lane_0' op incompatible distribution dimensions from 'vector<128xi32>' to 'vector<4xi32>'}}
  vector_ext.warp_execute_on_lane_0(%laneid)[64]
  args(%v0 : vector<4xi32>) {
   ^bb0(%arg0 : vector<128xi32>) :
  }
  return
}

// -----

func @warp_2_distributed_dims(%laneid: index) {
  // expected-error@+1 {{'vector_ext.warp_execute_on_lane_0' op expected only one dimension to be distributed from 'vector<128x128xi32>' to 'vector<4x4xi32>'}}
  %2 = vector_ext.warp_execute_on_lane_0(%laneid)[32] -> (vector<4x4xi32>) {
    %0 = arith.constant dense<2>: vector<128x128xi32>
    vector_ext.yield %0 : vector<128x128xi32>
  }
  return
}

// -----

func @warp_mismatch_rank(%laneid: index) {
  // expected-error@+1 {{'vector_ext.warp_execute_on_lane_0' op expected distributed vectors to have same rank and element type.}}
  %2 = vector_ext.warp_execute_on_lane_0(%laneid)[32] -> (vector<4x4xi32>) {
    %0 = arith.constant dense<2>: vector<128xi32>
    vector_ext.yield %0 : vector<128xi32>
  }
  return
}

// -----

func @warp_mismatch_rank(%laneid: index) {
  // expected-error@+1 {{'vector_ext.warp_execute_on_lane_0' op expected vector type for distributed operands.}}
  %2 = vector_ext.warp_execute_on_lane_0(%laneid)[32] -> (i32) {
    %0 = arith.constant dense<2>: vector<128xi32>
    vector_ext.yield %0 : vector<128xi32>
  }
  return
}

