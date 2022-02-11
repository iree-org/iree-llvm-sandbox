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
#include <sys/time.h>
#include <unistd.h>
/** Additional runtime functions used in Alp */

/// Print time (passed as a double constant)
extern "C" void print_time(double time_s) {
  fprintf(stdout, "%lf secs\n", time_s);
}

/// Print the pid of the current application (for profiling purposes)
extern "C" void print_pid() {
  int pid = getpid();
  fprintf(stdout, "pid: %i\n", pid);
}

/// Prints GFLOPS rating.
extern "C" void print_flops(double flops) {
  fprintf(stdout, "%lf GFLOPS\n", flops / 1.0E9);
}

extern "C" void printF32(float f) { fprintf(stdout, "%g", f); }
extern "C" void printNewline() { fputc('\n', stdout); }

/// Returns the number of seconds since Epoch 1970-01-01 00:00:00 +0000 (UTC).
extern "C" double rtclock() {
#ifndef _WIN32
  struct timeval tp;
  int stat = gettimeofday(&tp, NULL);
  if (stat != 0)
    fprintf(stdout, "Error returning time from gettimeofday: %d\n", stat);
  return (tp.tv_sec + tp.tv_usec * 1.0e-6);
#else
  fprintf(stderr, "Timing utility not implemented on Windows\n");
  return 0.0;
#endif // _WIN32
}