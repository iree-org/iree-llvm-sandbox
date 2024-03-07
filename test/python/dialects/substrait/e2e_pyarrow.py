# RUN: %PYTHON %s | FileCheck %s

import pyarrow as pa
import pyarrow.lib
import pyarrow.substrait

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
  plan = ir.Module.parse('''
    substrait.plan version 0 : 42 : 1 {
      relation {
        %0 = named_table @t as ["a", "b"] : tuple<si32, si32>
        yield %0 : tuple<si32, si32>
      }
    }
  ''')

  def table_provider(names, schema):
    if names != ["t"]:
      raise Exception("Unrecognized table name")
    columns = {"a": [1, 2, 3], "b": [7, 8, 9]}
    schema = pa.schema([('a', pa.int32()), ('b', pa.int32())])
    return pa.Table.from_pydict(columns, schema=schema)

  # Export MLIR to protobuf.
  pb_plan = ss.to_binpb(plan.operation)

  # Execute in pyrrow and print result.
  reader = pa.substrait.run_query(pa.lib.tobytes(pb_plan),
                                  table_provider=table_provider)
  print(reader.read_all())
  # CHECK-NEXT:          pyarrow.Table
  # CHECK-NEXT:          a: int32
  # CHECK-NEXT:          b: int32
  # CHECK-NEXT:          ----
  # CHECK-NEXT{LITERAL}: a: [[1,2,3]]
  # CHECK-NEXT{LITERAL}: b: [[7,8,9]]
