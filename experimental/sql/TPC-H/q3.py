import ibis


def get_ibis_query(MKTSEGMENT="BUILDING", DATE="1995-03-15"):
  from tpc_h_tables import lineitem, customer, orders

  q = customer.join(orders, customer.c_custkey == orders.o_custkey)
  q = q.join(lineitem, lineitem.l_orderkey == orders.o_orderkey)
  q = q.filter(
      [q.c_mktsegment == MKTSEGMENT, q.o_orderdate < DATE, q.l_shipdate > DATE])
  qg = q.group_by([q.l_orderkey, q.o_orderdate, q.o_shippriority])
  q = qg.aggregate(revenue=(q.l_extendedprice * (1 - q.l_discount)).sum())
  q = q.sort_by([ibis.desc(q.revenue), q.o_orderdate])
  q = q.limit(10)

  return q
