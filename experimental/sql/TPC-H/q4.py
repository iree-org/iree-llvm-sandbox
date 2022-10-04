from utils import add_date


def get_ibis_query(DATE="1993-07-01"):
  from tpc_h_tables import orders, lineitem
  cond = (lineitem.l_orderkey == orders.o_orderkey) & (lineitem.l_commitdate <
                                                       lineitem.l_receiptdate)
  q = orders.filter([
      cond.any(),
      orders.o_orderdate >= DATE,
      orders.o_orderdate < add_date(DATE, dm=3),
  ])
  q = q.group_by([orders.o_orderpriority])
  q = q.aggregate(order_count=orders.count())
  q = q.sort_by([orders.o_orderpriority])
  return q
