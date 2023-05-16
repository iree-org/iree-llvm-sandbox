// Check that tt dialect is available in structured-opt.
// RUN: structured-opt %s | FileCheck %s

// CHECK-LABEL: @hello_world
tt.func private @hello_world()
