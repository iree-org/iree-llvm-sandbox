// RUN: structured-opt -verify-diagnostics -split-input-file %s

// Test error if providing too many names (1 name for 0 fields).
substrait.plan version 0 : 42 : 1 {
  relation {
    // expected-error@+2 {{mismatching 'field_names' (["a"]) and result type ('tuple<>')}}
    // expected-note@+1 {{too many field names provided}}
    %0 = named_table @t1 as ["a"] : tuple<>
    yield %0 : tuple<>
  }
}

// -----

// Test error if providing too few names (0 names for 1 field).
substrait.plan version 0 : 42 : 1 {
  relation {
    // expected-error@+2 {{mismatching 'field_names' ([]) and result type ('tuple<si32>')}}
    // expected-error@+1 {{not enough field names provided}}
    %0 = named_table @t1 as [] : tuple<si32>
    yield %0 : tuple<si32>
  }
}


// -----

// Test error if providing duplicate field names in the same nesting level.
substrait.plan version 0 : 42 : 1 {
  relation {
    // expected-error@+2 {{mismatching 'field_names' (["a", "a"]) and result type ('tuple<si32, si32>')}}
    // expected-error@+1 {{duplicate field name: 'a'}}
    %0 = named_table @t1 as ["a", "a"] : tuple<si32, si32>
    yield %0 : tuple<si32, si32>
  }
}
