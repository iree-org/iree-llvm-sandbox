import ibis


def get_ibis_query(MKTSEGMENT="BUILDING", DATE="1995-03-15"):
  from tpc_h_tables import lineitem, customer, orders

  q = customer.join(orders, customer.c_custkey == orders.o_custkey)
  q = q.join(lineitem, lineitem.l_orderkey == orders.o_orderkey)
  q = q.filter(
      [q.c_mktsegment == MKTSEGMENT, q.o_orderdate < DATE, q.l_shipdate > DATE])
  proj = q.projection(
      q.columns + [(q.l_extendedprice *
                    (ibis.literal(1, "int64") - q.l_discount)).name("revenue")])
  q = proj.group_by([proj.l_orderkey, proj.o_orderdate, proj.o_shippriority
                    ]).aggregate(proj.revenue.sum().name('revenue'))
  q = q.sort_by([ibis.desc(q.revenue), q.o_orderdate])
  q = q.limit(10)

  return q
