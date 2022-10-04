import ibis
from decimal import Decimal
import numpy as np
from utils import add_date


def get_ibis_query(DELTA=90, DATE="1998-12-01"):
  from tpc_h_tables import lineitem

  t = lineitem

  interval = add_date(DATE, dd=-1 * DELTA)
  q = t.filter(t.l_shipdate <= interval)
  discount_price = t.l_extendedprice * (1 - t.l_discount)
  charge = discount_price * (1 + t.l_tax)
  q = q.group_by(["l_returnflag", "l_linestatus"])
  q = q.aggregate(
      sum_qty=t.l_quantity.sum(),
      sum_base_price=t.l_extendedprice.sum(),
      sum_disc_price=discount_price.sum(),
      sum_charge=charge.sum(),
      avg_qty=t.l_quantity.mean(),
      avg_price=t.l_extendedprice.mean(),
      avg_disc=t.l_discount.mean(),
      count_order=t.count(),
  )
  q = q.sort_by(["l_returnflag", "l_linestatus"])
  return q
