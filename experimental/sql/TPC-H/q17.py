def get_ibis_query(BRAND="Brand#23", CONTAINER="MED BOX"):
  from tpc_h_tables import lineitem, part

  q = lineitem.join(part, part.p_partkey == lineitem.l_partkey)

  innerq = lineitem
  innerq = innerq.filter([innerq.l_partkey == q.p_partkey])

  q = q.filter([
      q.p_brand == BRAND,
      q.p_container == CONTAINER,
      q.l_quantity < (0.2 * innerq.l_quantity.mean()),
  ])
  q = q.aggregate(avg_yearly=q.l_extendedprice.sum() / 7.0)
  return q
