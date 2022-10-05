import ibis


def get_ibis_query(
    QUANTITY1=ibis.literal(1, "int64"),
    QUANTITY2=ibis.literal(10, "int64"),
    QUANTITY3=ibis.literal(20, "int64"),
    BRAND1="Brand#12",
    BRAND2="Brand#23",
    BRAND3="Brand#34",
):
  from tpc_h_tables import lineitem, part
  q = lineitem.join(part, part.p_partkey == lineitem.l_partkey)

  q1 = ((q.p_brand == BRAND1) & (q.p_container.isin(
      ("SM CASE", "SM BOX", "SM PACK", "SM PKG"))) &
        (q.l_quantity >= QUANTITY1) &
        (q.l_quantity <= QUANTITY1 + ibis.literal(10, "int64")) &
        (q.p_size.between(1, 5)) & (q.l_shipmode.isin(
            ("AIR", "AIR REG"))) & (q.l_shipinstruct == "DELIVER IN PERSON"))

  q2 = ((q.p_brand == BRAND2) & (q.p_container.isin(
      ("MED BAG", "MED BOX", "MED PKG", "MED PACK"))) &
        (q.l_quantity >= QUANTITY2) &
        (q.l_quantity <= QUANTITY2 + ibis.literal(10, "int64")) &
        (q.p_size.between(1, 10)) & (q.l_shipmode.isin(
            ("AIR", "AIR REG"))) & (q.l_shipinstruct == "DELIVER IN PERSON"))

  q3 = ((q.p_brand == BRAND3) & (q.p_container.isin(
      ("LG CASE", "LG BOX", "LG PACK", "LG PKG"))) &
        (q.l_quantity >= QUANTITY3) &
        (q.l_quantity <= QUANTITY3 + ibis.literal(10, "int64")) &
        (q.p_size.between(1, 15)) & (q.l_shipmode.isin(
            ("AIR", "AIR REG"))) & (q.l_shipinstruct == "DELIVER IN PERSON"))

  q = q.filter([q1 | q2 | q3])
  proj = q.projection(
      (q.l_extendedprice *
       (ibis.literal(1, "int64") - q.l_discount)).name('revenue'))
  q = proj.aggregate(proj.revenue.sum())
  return q
