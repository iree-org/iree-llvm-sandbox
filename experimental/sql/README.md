# Prototype SQL dialect(s)

This folder contains a simple stand-alone project that implements a lowering pipeline from the [ibis frontend](https://ibis-project.org/) to the iterator dialect through [xDSL](https://github.com/xdslproject/xdsl).

## Requirements

To use, first install the dependencies:

```bash
pip install -r requirements.txt
```

## Execute a testcase

Run

```bash
python simple.py
```

This prints three representations of the query `table.filter(table['a'] == 'AS')`:

- the ibis representation
- the same representation mirrored in xDSL
- the query in the relational dialect

## Execute filecheck tests

First, add the path of this subproject to the `PYTHONPATH` environment variable.

```bash
export PYTHONPATH=$PYHTONPATH:$(git rev-parse --show-toplevel)/experimental/sql
```

Then run:

```bash
lit test/
```
