import ibis

from utils import add_date


def get_ibis_query(NAME="ASIA", DATE="1994-01-01"):
  from tpc_h_tables import customer, lineitem, orders, supplier, nation, region

  q = customer
  q = q.join(orders, customer.c_custkey == orders.o_custkey)
  q = q.join(lineitem, lineitem.l_orderkey == orders.o_orderkey)
  q = q.join(supplier, lineitem.l_suppkey == supplier.s_suppkey)
  q = q.join(
      nation,
      (customer.c_nationkey == supplier.s_nationkey) &
      (supplier.s_nationkey == nation.n_nationkey),
  )
  q = q.join(region, nation.n_regionkey == region.r_regionkey)

  q = q.filter([
      q.r_name == NAME, q.o_orderdate >= DATE,
      q.o_orderdate < add_date(DATE, dy=1)
  ])
  revexpr = q.l_extendedprice * (ibis.literal(1, "int64") - q.l_discount)
  proj = q.projection(q.columns + [(revexpr).name('revenue')])
  q = proj.group_by([proj.n_name]).aggregate(revenue=proj.revenue.sum())
  q = q.sort_by([ibis.desc(q.revenue)])
  return q
