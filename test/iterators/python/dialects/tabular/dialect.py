# RUN: %PYTHON %s | FileCheck %s

from mlir_iterators.dialects import tabular as tab
from mlir_iterators.ir import Context, IntegerType


def run(f):
  print("\nTEST:", f.__name__)
  with Context():
    tab.register_dialect()
    f()
  return f


# CHECK-LABEL: TEST: testTabularViewType
@run
def testTabularViewType():
  i32 = IntegerType.get_signless(32)
  tabular_view = tab.TabularViewType.get([i32])
  # CHECK: !tabular.tabular_view<i32>
  print(tabular_view)
  # CHECK: i32
  print(tabular_view.get_column_type(0))
  # CHECK: 1
  print(tabular_view.get_num_column_types())
  # CHECK: tuple<i32>
  print(tabular_view.get_row_type())
