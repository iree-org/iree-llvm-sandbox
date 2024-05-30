// RUN: structured-translate -verify-diagnostics -split-input-file %s \
// RUN:   -substrait-to-protobuf

// Two subsequent `emit` ops: the export can't deal with that.

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : tuple<si1, si32>
    // expected-note@+1 {{op exported to 'input' message}}
    %1 = emit [1, 0] from %0 : tuple<si1, si32> -> tuple<si32, si1>
    // expected-error@+1 {{'substrait.emit' op has 'input' that already has 'emit' message (try running canonicalization?)}}
    %2 = emit [1, 0] from %1 : tuple<si32, si1> -> tuple<si1, si32>
    yield %2 : tuple<si1, si32>
  }
}
