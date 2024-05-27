// RUN: structured-opt -verify-diagnostics -split-input-file %s


// expected-error@+1 {{unsuited attribute for literal value: unit}}
%0 = substrait.literal unit
