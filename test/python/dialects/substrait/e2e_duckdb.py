# RUN: %PYTHON %s | FileCheck %s

import duckdb

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
  con = duckdb.connect()
  con.install_extension("substrait")
  con.load_extension("substrait")

  con.execute(query="CREATE TABLE t (a INT NOT NULL, b INT NOT NULL)")
  con.execute(query="INSERT INTO t VALUES (1, 7)")
  con.execute(query="INSERT INTO t VALUES (2, 8)")
  con.execute(query="INSERT INTO t VALUES (3, 9)")

  # Set up test plan in MLIR.
  plan = ir.Module.parse('''
    substrait.plan version 0 : 42 : 1 {
      relation as ["a", "b"] {
        %0 = named_table @t as ["a", "b"] : tuple<si32, si32>
        yield %0 : tuple<si32, si32>
      }
    }
  ''')

  # Export MLIR plan to protobuf.
  pb_plan = ss.to_binpb(plan.operation).encode()

  # Execute in duckdb and print result.
  query_result = con.from_substrait(proto=pb_plan)

  print(query_result.to_arrow_table())
  # CHECK-NEXT:          pyarrow.Table
  # CHECK-NEXT:          a: int32
  # CHECK-NEXT:          b: int32
  # CHECK-NEXT:          ----
  # CHECK-NEXT{LITERAL}: a: [[1,2,3]]
  # CHECK-NEXT{LITERAL}: b: [[7,8,9]]
