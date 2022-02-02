// RUN: mlir-proto-opt %s -split-input-file -verify-diagnostics

linalg_transform.sequence {
  // expected-error@below {{result #0 has more than one use}}
  %0 = tile when @match
  // expected-note@below {{used here as operand #0}}
  tile %0
  // expected-note@below {{used here as operand #0}}
  vectorize %0
}
