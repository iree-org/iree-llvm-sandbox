# RUN: %PYTHON %s | FileCheck %s

import ibis
from ibis_substrait.compiler import core as ibis_ss

from mlir_structured.dialects import substrait as ss
from mlir_structured import ir


def run(f):
  print("\nTEST:", f.__name__)
  with ir.Context(), ir.Location.unknown():
    ss.register_dialect()
    f()
  return f


# CHECK-LABEL: TEST: testNamedTable
@run
def testNamedTable():
  # Set up test table.
  table = ibis.table([("a", "int32"), ("b", "int32")], "t")

  # Create Substrait plan from Ibis expression.
  compiler = ibis_ss.SubstraitCompiler()
  pb_plan = compiler.compile(table)

  # Import into MLIR and print.
  plan = ss.from_binpb(pb_plan.SerializeToString())
  print(plan)

  # CHECK-NEXT: module
  # CHECK-NEXT:   substrait.plan version {{.*}} producer "ibis-substrait" {
  # CHECK-NEXT:     relation as ["a", "b"] {
  # CHECK-NEXT:       %[[V0:.*]] = named_table @t as ["a", "b"] : tuple<si32, si32>
  # CHECK-NEXT:       yield %[[V0]] : tuple<si32, si32>
