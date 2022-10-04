import ibis


def get_ibis_query(WORD1="special", WORD2="requests"):
  from tpc_h_tables import customer, orders
  innerq = customer
  innerq = innerq.left_join(
      orders,
      (customer.c_custkey == orders.o_custkey) &
      ~orders.o_comment.like(f"%{WORD1}%{WORD2}%"),
  )
  innergq = innerq.group_by([innerq.c_custkey])
  innerq = innergq.aggregate(c_count=innerq.o_orderkey.count())

  gq = innerq.group_by([innerq.c_count])
  q = gq.aggregate(custdist=innerq.count())

  q = q.sort_by([ibis.desc(q.custdist), ibis.desc(q.c_count)])
  return
