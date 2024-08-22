// RUN: structured-translate -substrait-to-protobuf --split-input-file %s \
// RUN: | FileCheck %s

// RUN: structured-translate -substrait-to-protobuf %s \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | structured-translate -protobuf-to-substrait \
// RUN:   --split-input-file="# -----" --output-split-marker="// ""-----" \
// RUN: | structured-translate -substrait-to-protobuf \
// RUN:   --split-input-file --output-split-marker="# -----" \
// RUN: | FileCheck %s

// CHECK-LABEL: version {
// CHECK-DAG:     minor_number: 42
// CHECK-DAG:     patch_number: 1
// CHECK-DAG:     git_hash: "hash"
// CHECK-DAG:     producer: "producer"
// CHECK-NEXT:  }
substrait.plan
  version 0 : 42 : 1
  git_hash "hash"
  producer "producer"
  {}

// -----

// CHECK:      relations {
// CHECK-NEXT:   root {
// CHECK-NEXT:     input {
// CHECK-NEXT:       read {
// CHECK:              named_table {
// CHECK-NEXT:           names
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     names: "x"
// CHECK-NEXT:     names: "y"
// CHECK-NEXT:     names: "z"
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: version

substrait.plan version 0 : 42 : 1 {
  relation as ["x", "y", "z"] {
    %0 = named_table @t as ["a", "b", "c"] : tuple<si32, tuple<si32>>
    yield %0 : tuple<si32, tuple<si32>>
  }
}

// -----

// CHECK:      extension_uris {
// CHECK-NEXT:   extension_uri_anchor: 1
// CHECK-NEXT:   uri: "http://url.1/with/extensions.yml"
// CHECK:      extension_uris {
// CHECK-NEXT:   extension_uri_anchor: 2
// CHECK-NEXT:   uri: "http://url.2/with/extensions.yml"
// CHECK:      extension_uris {
// CHECK-NEXT:   extension_uri_anchor: 42
// CHECK-NEXT:   uri: "http://url.42/with/extensions.yml"
// CHECK:      extension_uris {
// CHECK-NEXT:   uri: "http://some.url/with/extensions.yml"
// CHECK:      extension_uris {
// CHECK-NEXT:   extension_uri_anchor: 3
// CHECK-NEXT:   uri: "http://url.foo/with/extensions.yml"
// CHECK:      extension_uris {
// CHECK-NEXT:   extension_uri_anchor: 4
// CHECK-NEXT:   uri: "http://url.bar/with/extensions.yml"
// CHECK:      extensions {
// CHECK-NEXT:   extension_function {
// CHECK-NEXT:     extension_uri_reference: 1
// CHECK-NEXT:     function_anchor: 1
// CHECK-NEXT:     name: "func1"
// CHECK:      extensions {
// CHECK-NEXT:   extension_function {
// CHECK-NEXT:     extension_uri_reference: 42
// CHECK-NEXT:     function_anchor: 42
// CHECK-NEXT:     name: "func42"
// CHECK:      extensions {
// CHECK-NEXT:   extension_type {
// CHECK-NEXT:     extension_uri_reference: 2
// CHECK-NEXT:     type_anchor: 1
// CHECK-NEXT:     name: "type1"
// CHECK:      extensions {
// CHECK-NEXT:   extension_type {
// CHECK-NEXT:     extension_uri_reference: 2
// CHECK-NEXT:     type_anchor: 42
// CHECK-NEXT:     name: "type42"
// CHECK:      extensions {
// CHECK-NEXT:   extension_type_variation {
// CHECK-NEXT:     extension_uri_reference: 1
// CHECK-NEXT:     type_variation_anchor: 1
// CHECK-NEXT:     name: "typevar1"
// CHECK:      extensions {
// CHECK-NEXT:   extension_type_variation {
// CHECK-NEXT:     extension_uri_reference: 1
// CHECK-NEXT:     type_variation_anchor: 42
// CHECK-NEXT:     name: "typevar2"

substrait.plan version 0 : 42 : 1 {
  extension_uri @extension_uri.1 at "http://url.1/with/extensions.yml"
  extension_uri @extension_uri.2 at "http://url.2/with/extensions.yml"
  extension_uri @extension_uri.42 at "http://url.42/with/extensions.yml"
  extension_uri @extension at "http://some.url/with/extensions.yml"
  extension_uri @extension_uri.foo at "http://url.foo/with/extensions.yml"
  extension_uri @extension_uri.bar at "http://url.bar/with/extensions.yml"
  extension_function @extension_function.1 at @extension_uri.1["func1"]
  extension_function @extension_function.42 at @extension_uri.42["func42"]
  extension_type @extension_type.1 at @extension_uri.2["type1"]
  extension_type @extension_type.42 at @extension_uri.2["type42"]
  extension_type_variation @extension_type_variation.1 at @extension_uri.1["typevar1"]
  extension_type_variation @extension_type_variation.42 at @extension_uri.1["typevar2"]
}

// -----


// CHECK:      extension_uris {
// CHECK-NEXT:   uri: "http://some.url/with/extensions.yml"
// CHECK:      extension_uris {
// CHECK-NEXT:   extension_uri_anchor: 1
// CHECK-NEXT:   uri: "http://other.url/with/more/extensions.yml"

substrait.plan version 0 : 42 : 1 {
  extension_uri @extension at "http://some.url/with/extensions.yml"
  // If not handled carefully, parsing this symbol into an anchor may clash.
  extension_uri @extension_uri.0 at "http://other.url/with/more/extensions.yml"
}
