# Instructions

Tests in this directory assume that IREE has been built with python extensions.
Assuming the IREE build directory is ${IREE_BUILD_DIR}, one would typically
export the `PYTHONPATH` command in `${IREE_BUILD_DIR}/.env`.

We reproduce it here:

```
PYTHONPATH=${PYTHONPATH}:${IREE_BUILD_DIR}/compiler/bindings/python:${IREE_BUILD_DIR}/runtime/bindings/python \
python <filename.py>
```
