// RUN: structured-translate -verify-diagnostics -split-input-file %s \
// RUN:   -substrait-to-protobuf

// Two different `emit` consumers: can't export into a single `Emit` message.

substrait.plan version 0 : 42 : 1 {
  relation {
    // expected-error@+1 {{'substrait.named_table' op is consumed by different emit ops (try running canonicalization and/or CSE)}}
    %0 = named_table @t1 as ["a", "b"] : tuple<si1, si32>
    %1 = emit [1, 0] from %0 : tuple<si1, si32> -> tuple<si32, si1>
    %2 = emit [0, 1] from %0 : tuple<si1, si32> -> tuple<si1, si32>
    yield %1 : tuple<si32, si1>
  }
}

// -----

// One `emit` consumer, one other consumer: can't export to a single `Emit`
// message.

substrait.plan version 0 : 42 : 1 {
  relation {
    // expected-error@+1 {{'substrait.named_table' op is consumed by different emit ops (try running canonicalization and/or CSE)}}
    %0 = named_table @t1 as ["a", "b"] : tuple<si1, si32>
    %1 = emit [1, 0] from %0 : tuple<si1, si32> -> tuple<si32, si1>
    yield %0 : tuple<si1, si32>
  }
}

// -----

// Two subsequent `emit` ops: the export can't deal with that.

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a", "b"] : tuple<si1, si32>
    %1 = emit [1, 0] from %0 : tuple<si1, si32> -> tuple<si32, si1>
    // expected-error@+1 {{'substrait.emit' op with input produced by 'substrait.emit' op not supported for export (try running canonicalization)}}
    %2 = emit [1, 0] from %1 : tuple<si32, si1> -> tuple<si1, si32>
    yield %2 : tuple<si1, si32>
  }
}
