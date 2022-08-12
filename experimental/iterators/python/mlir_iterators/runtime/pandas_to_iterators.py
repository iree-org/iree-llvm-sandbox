import ctypes

import numpy as np
import pandas as pd


def make_columnar_batch_descriptor(column_types: list[object]):
  '''
  Creates an empty instance a ctype.Structure corresponding to the
  LLVMSTructType that the columnar batch type with the given column types lowers
  to. The column types must be provided as ctypes.
  '''

  class ColumnarBatchDescriptor(ctypes.Structure):
    '''A descriptor of a columnar batch with columns of a particular type.'''

    _fields_ = [('num_elements', ctypes.c_longlong)] \
      + [('column' + str(i), ctypes.POINTER(dtype))
         for i, dtype in enumerate(column_types)]

  return ColumnarBatchDescriptor()


def to_columnar_batch_descriptor(df: pd.DataFrame):
  '''
  Converts the given DataFrame to an instance of ctype.Structure equivalent to
  what an instance of a corresponding iterators.ColumnarBatchType would get
  lowered to by IteratorsToLLVM.
  '''

  dtypes = [np.ctypeslib.as_ctypes_type(t) for t in df.dtypes]
  descriptor = make_columnar_batch_descriptor(dtypes)
  descriptor.num_elements = ctypes.c_longlong(len(df.index))
  for i, (dtype, col) in enumerate(zip(dtypes, df.columns)):
    setattr(descriptor, 'column' + str(i),
            df[col].values.ctypes.data_as(ctypes.POINTER(dtype)))
  return descriptor
