import datetime
from dateutil.relativedelta import relativedelta


def add_date(datestr, dy=0, dm=0, dd=0):
  dt = datetime.date.fromisoformat(datestr)
  dt += relativedelta(years=dy, months=dm, days=dd)
  return dt.isoformat()
