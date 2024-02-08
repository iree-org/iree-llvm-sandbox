// This file tests the tests infrastructure. It can be removed once we have any
// test case that actually tests something.

// RUN: jasc-opt %s -jasc-memcpy-to-gpu-dialect \
// RUN:   | jasc-opt \
// RUN:   | FileCheck %s

// CHECK: module
