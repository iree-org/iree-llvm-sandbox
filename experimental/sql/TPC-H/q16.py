import ibis

ibis.options.sql.default_limit = 100000


def get_ibis_query(BRAND="Brand#45",
                   TYPE="MEDIUM POLISHED",
                   SIZES=(49, 14, 23, 45, 19, 3, 36, 9)):
  from tpc_h_tables import part, partsupp, supplier

  q = partsupp.join(part, part.p_partkey == partsupp.ps_partkey)
  q = q.filter([
      q.p_brand != BRAND,
      ~q.p_type.like(f"{TYPE}%"),
      q.p_size.isin(SIZES),
      ~q.ps_suppkey.isin(
          supplier.filter([supplier.s_comment.like("%Customer%Complaints%")
                          ]).s_suppkey),
  ])
  gq = q.groupby([q.p_brand, q.p_type, q.p_size])
  q = gq.aggregate(supplier_cnt=q.ps_suppkey.nunique())
  q = q.sort_by([ibis.desc(q.supplier_cnt), q.p_brand, q.p_type, q.p_size])
  return
