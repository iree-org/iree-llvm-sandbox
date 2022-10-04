import ibis
from decimal import Decimal
import numpy as np

lineitem = ibis.table({
    "l_orderkey": "int64",
    "l_partkey": "int64",
    "l_suppkey": "int64",
    "l_linenumber": "int64",
    "l_quantity": "int64",
    "l_extendedprice": "decimal(32, 2)",
    "l_discount": "decimal(32, 2)",
    "l_tax": "decimal(32, 2)",
    "l_returnflag": "str",
    "l_linestatus": "str",
    "l_shipdate": "timestamp",
    "l_commitdate": "timestamp",
    "l_receiptdate": "timestamp",
    "l_shipinstruct": "str",
    "l_shipmode": "str",
    "l_comment": "str"
})

region = ibis.table({
    "r_regionkey": "int64",
    "r_name": "str",
    "r_comment": "str"
})

nation = ibis.table({
    "n_nationkey": "int64",
    "n_name": "str",
    "n_regionkey": "int64",
    "n_comment": "str"
})

part = ibis.table({
    "p_partkey": "int64",
    "p_name": "str",
    "p_mfgr": "str",
    "p_brand": "str",
    "p_type": "str",
    "p_size": "int64",
    "p_container": "str",
    "p_retailprice": "decimal(32, 2)",
    "p_comment": "str"
})

supplier = ibis.table({
    "s_suppkey": "int64",
    "s_name": "str",
    "s_address": "str",
    "s_nationkey": "int64",
    "s_phone": "str",
    "s_acctbal": "decimal(32,2)",
    "s_comment": "str"
})

partsupp = ibis.table({
    "ps_partkey": "int64",
    "ps_suppkey": "int64",
    "ps_availqty": "int64",
    "ps_supplycost": "decimal(32,2)",
    "ps_comment": "str"
})

customer = ibis.table({
    "c_custkey": "int64",
    "c_name": "str",
    "c_address": "str",
    "c_nationkey": "int64",
    "c_phone": "str",
    "c_acctbal": "decimal(32, 2)",
    "c_mktsegment": "str",
    "c_comment": "str"
})

orders = ibis.table({
    "o_orderkey": "int64",
    "o_custkey": "int64",
    "o_orderstatus": "str",
    "o_totalprice": "decimal(32, 2)",
    "o_orderdate": "timestamp",
    "o_orderpriority": "str",
    "o_clerk": "str",
    "o_shippriority": "int64",
    "o_comment": "str"
})
