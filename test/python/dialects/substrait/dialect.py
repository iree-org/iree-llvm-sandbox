# RUN: %PYTHON %s | FileCheck %s

from mlir_structured.dialects import substrait as ss
from mlir_structured import ir


def run(f):
  print("\nTEST:", f.__name__)
  with ir.Context(), ir.Location.unknown():
    ss.register_dialect()
    f()
  return f


# CHECK-LABEL: TEST: testSubstraitDialect
@run
def testSubstraitDialect():
  plan = ss.PlanOp(version=(0, 42, 1))
  print(plan)
  # CHECK: substrait.plan


# CHECK-LABEL: TEST: testPlanOp
@run
def testPlanOp():
  plan = ss.PlanOp(0, 42, 1)
  print(plan)
  # CHECK: substrait.plan version 0 : 42 : 1
  plan = ss.PlanOp(version=(0, 42, 1))
  print(plan)
  # CHECK: substrait.plan version 0 : 42 : 1


# CHECK-LABEL: TEST: testNamedTable
@run
def testNamedTable():
  plan = ss.PlanOp(version=(0, 42, 1))

  with ir.InsertionPoint(plan.body):
    plan_rel = ss.PlanRelOp()
    with ir.InsertionPoint(plan_rel.body):
      si32 = ir.IntegerType.get_signed(32)
      result_type = ir.TupleType.get_tuple([si32, si32])
      field_names = ir.ArrayAttr.get([ir.StringAttr.get(n) for n in ["a", "b"]])
      named_table = ss.NamedTableOp(result_type, "t", field_names)
      ss.YieldOp(named_table)

  print(plan)
  # CHECK: substrait.plan
  # CHECK: relation {
  # CHECK: named_table @t
