import q1
import q2
import q3
import q4
import q5
import q6
import q7
import q8
import q9
import q10
import q11
import q12
import q13
import q14
import q15
import q16
import q17
import q18
import q19
import q20
import q21
import q22

from src.ibis_frontend import ibis_to_xdsl
from src.ibis_to_alg import ibis_to_alg
from src.alg_to_ssa import alg_to_ssa
from src.ssa_to_impl import ssa_to_impl
from src.projection_pushdown import projection_pushdown
from src.fuse_proj_into_scan import fuse_proj_into_scan

from xdsl.ir import MLContext
from xdsl.printer import Printer

import sys
from io import StringIO


def compile(query):
  ctx = MLContext()
  mod = ibis_to_xdsl(ctx, query)
  ibis_to_alg(ctx, mod)
  #projection_pushdown(ctx, mod)
  #alg_to_ssa(ctx, mod)
  #ssa_to_impl(ctx, mod)
  #fuse_proj_into_scan(ctx, mod)

  return mod


def get_tpc_queries():
  queries = []
  queries.append(q1.get_ibis_query())
  #queries.append(q2.get_ibis_query())
  queries.append(q3.get_ibis_query())
  queries.append(q4.get_ibis_query())
  queries.append(q5.get_ibis_query())
  queries.append(q6.get_ibis_query())
  #queries.append(q7.get_ibis_query())
  #queries.append(q8.get_ibis_query())
  queries.append(q9.get_ibis_query())
  queries.append(q10.get_ibis_query())
  queries.append(q11.get_ibis_query())
  #queries.append(q12.get_ibis_query())
  #queries.append(q13.get_ibis_query())
  #queries.append(q14.get_ibis_query())
  #queries.append(q15.get_ibis_query())
  queries.append(q16.get_ibis_query())
  queries.append(q17.get_ibis_query())
  queries.append(q18.get_ibis_query())
  #queries.append(q19.get_ibis_query())
  #queries.append(q20.get_ibis_query())
  queries.append(q21.get_ibis_query())
  #queries.append(q22.get_ibis_query())
  return queries


def run():
  _stdout = sys.stdout
  sys.stdout = s = StringIO()
  for i, q in enumerate(get_tpc_queries()):
    print(i + 1)
    compile(q)
  l = s.getvalue().splitlines()
  sys.stdout = _stdout
  return l


def parse_data(f: str):
  hardness = []
  curr_hardness = []
  for line in f:
    l = line.strip()
    if l.isnumeric():
      hardness.append(list(dict.fromkeys(curr_hardness)))
      curr_hardness = []
    else:
      curr_hardness.append(l)
  hardness.append(list(dict.fromkeys(curr_hardness)))
  return hardness[1:]


def evaluate():
  lists = parse_data(run())
  handled = 0
  for i, l in enumerate(lists):
    if l == []:
      handled += 1
    else:
      print(i + 1)
      print(l)
  print()
  print(f"Handled: {handled}")


if __name__ == "__main__":
  run()
