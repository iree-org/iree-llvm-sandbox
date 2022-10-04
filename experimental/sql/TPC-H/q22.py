def get_ibis_query(COUNTRY_CODES=("13", "31", "23", "29", "30", "18", "17")):
  from tpc_h_tables import customer, orders

  q = customer.filter([
      customer.c_acctbal > 0.00,
      customer.c_phone.substr(0, 2).isin(COUNTRY_CODES),
  ])
  q = q.aggregate(avg_bal=customer.c_acctbal.mean())

  custsale = customer.filter([
      customer.c_phone.substr(0, 2).isin(COUNTRY_CODES),
      customer.c_acctbal > q.avg_bal,
      ~(orders.o_custkey == customer.c_custkey).any(),
  ])
  custsale = custsale[customer.c_phone.substr(0, 2).name("cntrycode"),
                      customer.c_acctbal]

  gq = custsale.group_by(custsale.cntrycode)
  outerq = gq.aggregate(numcust=custsale.count(),
                        totacctbal=custsale.c_acctbal.sum())

  return outerq.sort_by(outerq.cntrycode)
