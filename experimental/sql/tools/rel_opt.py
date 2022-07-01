#!/usr/bin/env python3
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
  from tools.IteratorsMLIRConverter import IteratorsMLIRConverter
  mlir_loaded = True
except ImportError:
  mlir_loaded = False

from xdsl.xdsl_opt_main import xDSLOptMain
from io import IOBase
from xdsl.dialects.builtin import ModuleOp

from src.ibis_frontend import ibis_to_xdsl
from src.ibis_to_alg import ibis_to_alg
from src.alg_to_ssa import alg_to_ssa
from src.ssa_to_impl import ssa_to_impl
from src.impl_to_iterators import impl_to_iterators
from dialects.ibis_dialect import Ibis
from dialects.rel_alg import RelationalAlg
from dialects.rel_ssa import RelSSA
from dialects.rel_impl import RelImpl
from dialects.iterators import Iterators


class RelOptMain(xDSLOptMain):

  def register_all_passes(self):
    self.available_passes['ibis-to-alg'] = ibis_to_alg
    self.available_passes['alg-to-ssa'] = alg_to_ssa
    self.available_passes['ssa-to-impl'] = ssa_to_impl
    self.available_passes['impl-to-iterators'] = impl_to_iterators

  def register_all_frontends(self):
    super().register_all_frontends()

    def parse_ibis(f: IOBase):
      import ibis
      import numpy as np
      import decimal

      def exec_then_eval(query: str):
        """
        Takes a (potentially multi-line) string that defines an ibis query with
        variables for subexpressions. The expression tree is built with the ith
        names of `names` corresponding to the ith table of `tables`. Finally,
        the result of the last line (modulo comments) is returned.
        """
        import ast

        _globals, _locals = {}, {}

        _locals["np"] = np
        _locals["ibis"] = ibis
        _locals["decimal"] = decimal

        ast_query = ast.parse(query)
        last = ast.Expression(ast_query.body.pop().value)
        exec(compile(ast_query, '<string>', mode='exec'), _globals, _locals)
        return eval(compile(last, '<string>', mode='eval'), _globals, _locals)

      query = f.read()
      res = exec_then_eval(query)
      return ibis_to_xdsl(self.ctx, res)

    self.available_frontends['ibis'] = parse_ibis

  def register_all_dialects(self):
    super().register_all_dialects()
    """Register all dialects that can be used."""
    ibis = Ibis(self.ctx)
    rel_Alg = RelationalAlg(self.ctx)
    rel_ssa = RelSSA(self.ctx)
    rel_impl = RelImpl(self.ctx)
    dataflow = Iterators(self.ctx)

  def register_all_targets(self):
    super().register_all_targets()

    def _output_mlir(prog: ModuleOp, output: IOBase):
      converter = IteratorsMLIRConverter(self.ctx)
      mlir_module = converter.convert_module(prog)
      print(mlir_module, file=output)

    if mlir_loaded:
      self.available_targets['mlir'] = _output_mlir


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
