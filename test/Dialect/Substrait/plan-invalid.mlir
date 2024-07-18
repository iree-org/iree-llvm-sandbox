// RUN: structured-opt -verify-diagnostics -split-input-file %s

// Test error if no symbol was found for `extension_function` op.
substrait.plan version 0 : 42 : 1 {
  // expected-error@+1 {{'substrait.extension_function' op refers to @extension, which is not a valid 'uri' op}}
  extension_function @function at @extension["somefunc"]
}

// -----

// Test error if no symbol was found for `extension_type` op.
substrait.plan version 0 : 42 : 1 {
  // expected-error@+1 {{'substrait.extension_type' op refers to @extension, which is not a valid 'uri' op}}
  extension_type @type at @extension["sometype"]
}

// -----

// Test error if no symbol was found for `extension_type_variation` op.
substrait.plan version 0 : 42 : 1 {
  // expected-error@+1 {{'substrait.extension_type_variation' op refers to @extension, which is not a valid 'uri' op}}
  extension_type_variation @type_var at @extension["sometypevar"]
}

// -----

// Test error if symbol was in the wrong scope.
substrait.extension_uri @extension at "http://some.url/with/extensions.yml"
substrait.plan version 0 : 42 : 1 {
  // expected-error@+1 {{'substrait.extension_function' op refers to @extension, which is not a valid 'uri' op}}
  extension_function @function at @extension["somefunc"]
}

// -----

// Test error if no symbol refers to an op of the wrong type.
substrait.plan version 0 : 42 : 1 {
  extension_uri @extension at "http://some.url/with/extensions.yml"
  extension_function @function.1 at @extension["somefunc"]
  // expected-error@+1 {{'substrait.extension_function' op refers to @function.1, which is not a valid 'uri' op}}
  extension_function @function.2 at @function.1["somefunc"]
}
