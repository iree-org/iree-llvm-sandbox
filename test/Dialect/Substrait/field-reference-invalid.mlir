// RUN: structured-opt -verify-diagnostics -split-input-file %s

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = filter %0 : tuple<si32> {
    ^bb0(%arg : tuple<si32>):
      // expected-error@+2 {{can't extract element from type 'si32'}}
      // expected-error@+1 {{mismatching position and type (position: array<i64: 0, 0>, type: 'tuple<si32>')}}
      %2 = field_reference %arg[0, 0] : tuple<si32>
      %3 = literal 0 : si1
      yield %3 : si1
    }
    yield %1 : tuple<si32>
  }
}

// -----

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    %1 = filter %0 : tuple<si32> {
    ^bb0(%arg : tuple<si32>):
      // expected-error@+2 {{2 is not a valid index for 'tuple<si32>'}}
      // expected-error@+1 {{mismatching position and type (position: array<i64: 2>, type: 'tuple<si32>')}}
      %2 = field_reference %arg[2] : tuple<si32>
      %3 = literal 0 : si1
      yield %3 : si1
    }
    yield %1 : tuple<si32>
  }
}
