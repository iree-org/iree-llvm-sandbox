import ibis
from utils import add_date


def get_ibis_query(DATE="1993-10-01"):
  from tpc_h_tables import customer, orders, lineitem, nation

  q = customer
  q = q.join(orders, customer.c_custkey == orders.o_custkey)
  q = q.join(lineitem, lineitem.l_orderkey == orders.o_orderkey)
  q = q.join(nation, customer.c_nationkey == nation.n_nationkey)

  q = q.filter([
      q.o_orderdate >= DATE,
      q.o_orderdate < add_date(DATE, dm=3),
      q.l_returnflag == "R",
  ])

  proj = q.projection(
      q.columns + [(q.l_extendedprice *
                    (ibis.literal(1, "int64") - q.l_discount)).name('revenue')])
  q = proj.group_by([
      q.c_custkey,
      q.c_name,
      q.c_acctbal,
      q.c_phone,
      q.n_name,
      q.c_address,
      q.c_comment,
  ]).aggregate(revenue=proj.revenue.sum())

  q = q.sort_by(ibis.desc(q.revenue))
  return q.limit(20)
