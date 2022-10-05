import ibis
from decimal import Decimal
import numpy as np
from utils import add_date


def get_ibis_query(DELTA=90, DATE="1998-12-01"):
  from tpc_h_tables import lineitem

  t = lineitem

  interval = add_date(DATE, dd=-1 * DELTA)
  q = t.filter(t.l_shipdate <= interval)
  discount_price = t.l_extendedprice * (ibis.literal(1, "int64") - t.l_discount)
  charge = discount_price * (ibis.literal(1, "int64") + t.l_tax)
  proj = q.projection(
      q.columns +
      [discount_price.name('discount_price'),
       charge.name('charge')])
  q = proj.group_by(["l_returnflag", "l_linestatus"])
  q = q.aggregate(
      sum_qty=t.l_quantity.sum(),
      sum_base_price=t.l_extendedprice.sum(),
      sum_disc_price=proj.discount_price.sum(),
      sum_charge=proj.charge.sum(),
      avg_qty=t.l_quantity.mean(),
      avg_price=t.l_extendedprice.mean(),
      avg_disc=t.l_discount.mean(),
      count_order=t.count(),
  )
  q = q.sort_by(["l_returnflag", "l_linestatus"])
  return q
