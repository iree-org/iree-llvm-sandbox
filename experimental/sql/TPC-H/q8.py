import ibis

from utils import add_date


def get_ibis_query(
    NATION="BRAZIL",
    REGION="AMERICA",
    TYPE="ECONOMY ANODIZED STEEL",
    DATE="1995-01-01",
):
  from tpc_h_tables import part, supplier, lineitem, orders, nation, customer, region
  n1 = nation
  n2 = n1.view()

  q = part
  q = q.join(lineitem, part.p_partkey == lineitem.l_partkey)
  q = q.join(supplier, supplier.s_suppkey == lineitem.l_suppkey)
  q = q.join(orders, lineitem.l_orderkey == orders.o_orderkey)
  q = q.join(customer, orders.o_custkey == customer.c_custkey)
  q = q.join(n1, customer.c_nationkey == n1.n_nationkey)
  q = q.join(region, n1.n_regionkey == region.r_regionkey)
  q = q.join(n2, supplier.s_nationkey == n2.n_nationkey)

  q = q[orders.o_orderdate.year().cast("string").name("o_year"),
        (lineitem.l_extendedprice * (1 - lineitem.l_discount)).name("volume"),
        n2.n_name.name("nation"), region.r_name, orders.o_orderdate,
        part.p_type,]

  q = q.filter([
      q.r_name == REGION,
      q.o_orderdate.between(DATE, add_date(DATE, dy=2, dd=-1)),
      q.p_type == TYPE,
  ])

  q = q.mutate(nation_volume=ibis.case().when(q.nation == NATION,
                                              q.volume).else_(0).end())
  gq = q.group_by([q.o_year])
  q = gq.aggregate(mkt_share=q.nation_volume.sum() / q.volume.sum())
  q = q.sort_by([q.o_year])
  return q
