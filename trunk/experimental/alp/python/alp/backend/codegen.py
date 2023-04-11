#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from .utils import print_command, run_command


def codegen(source, dest, scheduler="ilpmax"):
  cmd = ["$IREE_LLVM_SANDBOX_BUILD_DIR/bin/llc"]
  cmd.append(source)
  cmd.append("-filetype=asm")
  cmd.append("-O3")

  # Scheduling options
  if scheduler == "ilpmax":
    cmd.append("--misched=ilpmax")
  elif scheduler == "shuffle":
    cmd.append("--misched=shuffle")
  else:
    raise (ValueError("Invalid scheduler algorithm"))

  # Register allocator options
  cmd.append(f"-o {dest}")
  run_command(cmd)
