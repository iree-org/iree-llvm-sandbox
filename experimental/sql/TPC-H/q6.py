from utils import add_date


def get_ibis_query(DATE="1994-01-01", DISCOUNT=0.06, QUANTITY=24):
  from tpc_h_tables import lineitem
  q = lineitem
  discount_min = round(DISCOUNT - 0.01, 2)
  discount_max = round(DISCOUNT + 0.01, 2)
  q = q.filter([
      q.l_shipdate >= DATE,
      q.l_shipdate < add_date(DATE, dy=1),
      q.l_discount.between(discount_min, discount_max),
      q.l_quantity < QUANTITY,
  ])
  q = q.aggregate(revenue=(q.l_extendedprice * q.l_discount).sum())
  return
