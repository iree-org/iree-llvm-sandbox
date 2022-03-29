# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file contains a simple testcase, that defines a table and runs a selection query on it.

import ibis
import os
import sys

import pandas as pd

from xdsl.ir import MLContext
from xdsl.printer import Printer
from src.ibis_frontend import ibis_to_xdsl
from src.ibis_to_rel import ibis_dialect_to_relational

connection = ibis.pandas.connect({"t": pd.DataFrame({"a": ["AS", "EU", "NA"]})})

# Get the table.
table = connection.table('t')

# Define the query.
query = table.filter(table['a'] == 'AS')
print('Ibis AST: ' + '-' * 60)
print(query)

# Define an xDSL printer (xdsl needs a Printer class to print since printing requires keeping state).
p = Printer()

# Define a MLContext, which keeps track of which operations are registered.
ctx = MLContext()

# Translate the query to the xDSL mirrored dialect.
xdsl_query = ibis_to_xdsl(ctx, query)
xdsl_query.verify()
print('Ibis dialect: ' + '-' * 60)
p.print_op(xdsl_query)

# Rewriter the query to the relational dialect.
ibis_dialect_to_relational(ctx, xdsl_query)

print('Relational dialect: ' + '-' * 60)
p.print_op(xdsl_query)
