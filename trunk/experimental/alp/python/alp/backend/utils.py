#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import subprocess
import os
import numpy as np
from subprocess import PIPE, Popen


def run_command(cmd):
  output = subprocess.check_output(" ".join(cmd), shell=True)
  return output.decode("ascii")


def print_command(cmd):
  print(" ".join(cmd))


def run_and_save(cmd, original_ir, new_ir):
  out = run_command(cmd + [original_ir])
  f = open(f"{new_ir}", "w")
  # Save the command that generated the IR
  f.write("//" + " ".join(cmd + [original_ir]) + "\n")
  # Save the IR
  f.write(out)
  f.close()


def add_extension(fname, ext):
  orig_ext = os.path.splitext(fname)[1]
  newfilename = os.path.splitext(fname)[0] + "." + ext + orig_ext
  return newfilename


def parse(out):
  if isinstance(out, bytes):
    out = out.decode("utf-8")

  secs = 0
  flops = 0
  lines = out.split("\n")
  for l in lines:
    if not l:
      continue
    [a, b] = l.split()
    if b == "secs":
      secs = float(a)
    if b == "GFLOPS":
      flops = float(a)
  return (secs, flops)


def analytical_model(hw, Sdata):
  # Analyitical model for GEMM
  # https://www.cs.utexas.edu/users/flame/pubs/TOMS-BLIS-Analytical.pdf

  # Vector unit properties
  Nvec = hw["Nvec"]
  Lvfma = hw["Lvfma"]
  Nvfma = hw["Nvfma"]

  # Determine mr/nr
  K = Nvec * Nvfma * Lvfma
  mr = np.ceil((np.sqrt(K) / Nvec)) * Nvec
  nr = np.ceil(K / mr)

  # L1 properties
  SL1 = hw["SL"][0] * 1024
  WL1 = hw["WL"][0]

  # L2 properties
  SL2 = hw["SL"][1] * 1024
  WL2 = hw["WL"][1]

  if "CL" in hw:
    CL1 = hw["CL"][0]
    CL2 = hw["CL"][1]
    NL1 = SL1 / (WL1 * CL1)
    NL2 = SL2 / (WL2 * CL2)
  elif "NL" in hw:
    NL1 = hw["NL"][0]
    NL2 = hw["NL"][1]
    CL1 = SL1 / (WL1 * NL1)
    CL2 = SL2 / (WL2 * NL2)

  # if L3 properties are specified, then determine nc
  if hw["num_caches"] == 3:
    SL3 = hw["SL"][2] * 1024
    WL3 = hw["WL"][2]

    if "CL" in hw:
      CL3 = hw["CL"][2]
      NL3 = SL3 / (WL3 * CL3)
    elif "NL" in hw:
      NL3 = hw["NL"][2]
      CL3 = SL3 / (WL3 * NL3)

  # Determine kc
  CAr = np.floor((WL1 - 1) / (1 + nr / mr))
  kc = (CAr * NL1 * CL1) / (mr * Sdata)

  #  Determine mc
  CBr2 = np.ceil(nr * kc * Sdata / (NL2 * CL2))
  mc = (WL2 - 1 - CBr2) * NL2 * CL2 / (kc * Sdata)

  # Determine nc
  if hw["num_caches"] == 3:
    CAc3 = np.ceil(mc * kc * Sdata / (NL3 * CL3))
    nc = ((WL3 - CAc3 - 1) * NL3 * CL3) / (kc * Sdata)
  else:
    nc = -1

  return (mc, nc, kc, mr, nr)
