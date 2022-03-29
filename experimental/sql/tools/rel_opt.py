#!/usr/bin/env python3
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from xdsl.ir import MLContext
from xdsl.dialects.builtin import ModuleOp
from typing import Callable, Dict, List
from xdsl.xdsl_opt_main import xDSLOptMain
from io import IOBase

from src.ibis_frontend import ibis_to_xdsl
from src.ibis_to_rel import ibis_dialect_to_relational
from dialects.ibis_dialect import Ibis
from dialects.relational_dialect import Relational


class RelOptMain(xDSLOptMain):

  def register_all_passes(self):
    self.available_passes[
        'ibis-dialect-to-relational'] = ibis_dialect_to_relational

  def register_all_frontends(self):
    super().register_all_frontends()

    def parse_ibis(f: IOBase):
      import ibis
      import pandas as pd

      connection = ibis.pandas.connect(
          {"t": pd.DataFrame({"a": ["AS", "EU", "NA"]})})
      table = connection.table('t')
      f.readline()
      f.readline()
      query = f.readline()
      res = eval(query)

      return ibis_to_xdsl(self.ctx, res)

    self.available_frontends['ibis'] = parse_ibis

  def register_all_dialects(self):
    super().register_all_dialects()
    """Register all dialects that can be used."""
    ibis = Ibis(self.ctx)
    rel = Relational(self.ctx)


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
