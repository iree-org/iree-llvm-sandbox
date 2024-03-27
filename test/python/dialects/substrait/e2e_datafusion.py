# RUN: %PYTHON %s | FileCheck %s

import datafusion
from datafusion import substrait as dfss
import pyarrow as pa

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
  ctx = datafusion.SessionContext()
  columns = {"a": [1, 2, 3], "b": [7, 8, 9]}
  schema = pa.schema([('a', pa.int32()), ('b', pa.int32())])
  batch = pa.RecordBatch.from_pydict(columns, schema=schema)
  ctx.register_record_batches("t", [[batch]])

  # Set up test plan in MLIR.
  plan = ir.Module.parse('''
    substrait.plan version 0 : 42 : 1 {
      relation {
        %0 = named_table @t as ["a", "b"] : tuple<si32, si32>
        yield %0 : tuple<si32, si32>
      }
    }
  ''')

  # Export MLIR plan to protobuf.
  pb_plan = ss.to_binpb(plan.operation)
  pb_plan = pb_plan.encode('utf8')

  # Import plan in datafusion, execute, and print result.
  ss_plan = dfss.substrait.serde.deserialize_bytes(pb_plan)
  df_plan = dfss.substrait.consumer.from_substrait_plan(ctx, ss_plan)
  df = ctx.create_dataframe_from_logical_plan(df_plan)

  print(df.to_arrow_table())
  # CHECK-NEXT:          pyarrow.Table
  # CHECK-NEXT:          a: int32
  # CHECK-NEXT:          b: int32
  # CHECK-NEXT:          ----
  # CHECK-NEXT{LITERAL}: a: [[1,2,3]]
  # CHECK-NEXT{LITERAL}: b: [[7,8,9]]
