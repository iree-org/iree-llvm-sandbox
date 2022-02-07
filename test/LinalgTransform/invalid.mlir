// RUN: mlir-proto-opt %s -split-input-file -verify-diagnostics

linalg_transform.sequence {
  // expected-error@below {{result #0 has more than one use}}
  %0 = tile when @match
  // expected-note@below {{used here as operand #0}}
  tile %0
  // expected-note@below {{used here as operand #0}}
  vectorize %0
}

// -----

linalg_transform.sequence {
  // expected-error@below {{expects transpose paddings to be a permutation, found [2, 0]}}
  tile when @match {pad = true, transpose_paddings = [[0, 1], [2, 0]]}
}

// -----

linalg_transform.sequence {
  // expected-error@below {{"sizes" and "scalarize_dyn_dims" attributes are mutually exclusive}}
  tile when @match {sizes = [1,2,3], scalarize_dyn_dims = true}
}
