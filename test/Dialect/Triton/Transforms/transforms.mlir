// Check that transforms on tt dialect are available in structured-opt.
// RUN: structured-opt -triton-combine %s | FileCheck %s

// CHECK-LABEL: @hello_world
tt.func private @hello_world()
