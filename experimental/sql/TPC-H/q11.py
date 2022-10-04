import ibis


def get_ibis_query(NATION="GERMANY", FRACTION=0.0001):
  from tpc_h_tables import partsupp, supplier, nation

  q = partsupp
  q = q.join(supplier, partsupp.ps_suppkey == supplier.s_suppkey)
  q = q.join(nation, nation.n_nationkey == supplier.s_nationkey)

  q = q.filter([q.n_name == NATION])

  innerq = partsupp
  innerq = innerq.join(supplier, partsupp.ps_suppkey == supplier.s_suppkey)
  innerq = innerq.join(nation, nation.n_nationkey == supplier.s_nationkey)
  innerq = innerq.filter([innerq.n_name == NATION])
  innerq = innerq.aggregate(total=(innerq.ps_supplycost *
                                   innerq.ps_availqty).sum())

  gq = q.group_by([q.ps_partkey])
  q = gq.aggregate(value=(q.ps_supplycost * q.ps_availqty).sum())
  q = q.filter([q.value > innerq.total * FRACTION])
  q = q.sort_by(ibis.desc(q.value))
  return q
