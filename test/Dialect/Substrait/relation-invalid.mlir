// RUN: structured-opt -verify-diagnostics -split-input-file %s

// Test error if providing too many names (1 name for 0 fields).
substrait.plan version 0 : 42 : 1 {
  // expected-error@+2 {{'substrait.relation' op has mismatching 'field_names' (["x", "y"]) and result type ('tuple<si32>')}}
  // expected-note@+1 {{too many field names provided}}
  relation as ["x", "y"] {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    yield %0 : tuple<si32>
  }
}

// -----

// Test error if providing too few names (0 names for 1 field).
substrait.plan version 0 : 42 : 1 {
  // expected-error@+2 {{'substrait.relation' op has mismatching 'field_names' (["x"]) and result type ('tuple<si32, si32>')}}
  // expected-error@+1 {{not enough field names provided}}
  relation as ["x"] {
    %0 = named_table @t1 as ["a", "b"] : tuple<si32, si32>
    yield %0 : tuple<si32, si32>
  }
}


// -----

// Test error if providing duplicate field names in the same nesting level.
substrait.plan version 0 : 42 : 1 {
  // expected-error@+2 {{'substrait.relation' op has mismatching 'field_names' (["x", "x"]) and result type ('tuple<si32, si32>')}}
  // expected-error@+1 {{duplicate field name: 'x'}}
  relation as ["x", "x"] {
    %0 = named_table @t1 as ["a", "b"] : tuple<si32, si32>
    yield %0 : tuple<si32, si32>
  }
}

// -----

// Test error on wrong number of yielded values.
substrait.plan version 0 : 42 : 1 {
  // expected-error@+1 {{'substrait.relation' op must have 'body' region yielding one value (yields 2)}}
  relation {
    %0 = named_table @t1 as ["a", "b"] : tuple<si32, si32>
    yield %0, %0 : tuple<si32, si32>, tuple<si32, si32>
  }
}
