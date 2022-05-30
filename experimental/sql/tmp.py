import ibis
import pandas as pd

connection = ibis.pandas.connect({
    "t":
        pd.DataFrame({
            "a": ["AS", "EU", "NA"],
            "b": [1, 2, 3],
            "c": [3, 1, 18]
        }),
    "u":
        pd.DataFrame({
            "b": [1, 2, 3],
        })
})
table = connection.table('t')
sum_table = connection.table('u')

proj = (table['b'] * table['c']).name('d')
proj2 = (table['b'] * table['c'])
table2 = table[(table['b'] * table['c']).name('d')]

print(table2.op().inputs[1][0].get_name())
print(table2.op())
