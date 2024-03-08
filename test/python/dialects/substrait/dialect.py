# RUN: %PYTHON %s | FileCheck %s

from mlir_structured.dialects import substrait as ss
from mlir_structured.ir import Context, Location


def run(f):
  print("\nTEST:", f.__name__)
  with Context(), Location.unknown():
    ss.register_dialect()
    f()
  return f


# CHECK-LABEL: TEST: testSubstraitDialect
@run
def testSubstraitDialect():
  plan = ss.PlanOp(0, 42, 1)
  print(plan)
  # CHECK: substrait.plan
