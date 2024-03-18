// RUN: structured-opt -verify-diagnostics -split-input-file %s

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    // expected-error@+1 {{'substrait.filter' op  must have 'condition' region yielding 'si1' (yields 'si32')}}
    %1 = filter %0 : tuple<si32> {
    ^bb0(%arg : tuple<si32>):
      %2 = literal 42 : si32
      yield %2 : si32
    }
    yield %1 : tuple<si32>
  }
}

// -----

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    // expected-error@+1 {{'substrait.filter' op must have 'condition' region taking 'tuple<si32>' as argument (takes no arguments)}}
    %1 = filter %0 : tuple<si32> {
      %2 = literal 0 : si1
      yield %2 : si1
    }
    yield %1 : tuple<si32>
  }
}

// -----

substrait.plan version 0 : 42 : 1 {
  relation {
    %0 = named_table @t1 as ["a"] : tuple<si32>
    // expected-error@+1 {{'substrait.filter' op must have 'condition' region taking 'tuple<si32>' as argument (takes 'tuple<>')}}
    %1 = filter %0 : tuple<si32> {
    ^bb0(%arg : tuple<>):
      %2 = literal 0 : si1
      yield %2 : si1
    }
    yield %1 : tuple<si32>
  }
}
