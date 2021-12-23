//===-- alp_runtime.cpp - Alp Extended Runtime ------*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <cinttypes>
#include <cstdio>
#include <string.h>
#include <unistd.h>
/** Additional runtime functions used in Alp */

/// Print time (passed as a double constant)
extern "C" void print_time(double time_s) {
  fprintf(stderr, "%lf secs\n", time_s);
}

/// Print the pid of the current application (for profiling purposes)
extern "C" void print_pid() {
  int pid = getpid();
  fprintf(stderr, "pid: %i\n", pid);
}
