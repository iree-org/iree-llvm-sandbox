#!/usr/bin/env python3
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from xdsl.xdsl_opt_main import xDSLOptMain

from dialects.ibis_dialect import Ibis


class RelOptMain(xDSLOptMain):

  def register_all_dialects(self):
    super().register_all_dialects()
    """Register all dialects that can be used."""
    ibis = Ibis(self.ctx)


def __main__():
  rel_main = RelOptMain()
  try:
    module = rel_main.parse_input()
    rel_main.apply_passes(module)
  except Exception as e:
    print(e)
    exit(0)

  contents = rel_main.output_resulting_program(module)
  rel_main.print_to_output_stream(contents)


if __name__ == "__main__":
  __main__()
