import ctypes

import numpy as np
import pandas as pd


def make_tabular_view_descriptor(column_types: list[object]):
  '''
  Creates an empty instance a ctype.Structure corresponding to the
  LLVMSTructType that the tabular view type with the given column types lowers
  to. The column types must be provided as ctypes.
  '''

  class TabularViewDescriptor(ctypes.Structure):
    '''A descriptor of a tabular view with columns of a particular type.'''

    _fields_ = [('num_elements', ctypes.c_longlong)] \
      + [('column' + str(i), ctypes.POINTER(dtype))
         for i, dtype in enumerate(column_types)]

  return TabularViewDescriptor()


def to_tabular_view_descriptor(df: pd.DataFrame):
  '''
  Converts the given DataFrame to an instance of ctype.Structure equivalent to
  what an instance of a corresponding tabular.TabularViewType would get lowered
  to by TabularToLLVM. This is zero-copy: the conversion constis of extracting
  the pointers from the pandas data structure and wrapping those.
  '''

  dtypes = [np.ctypeslib.as_ctypes_type(t) for t in df.dtypes]
  descriptor = make_tabular_view_descriptor(dtypes)
  descriptor.num_elements = ctypes.c_longlong(len(df.index))
  for i, (dtype, col) in enumerate(zip(dtypes, df.columns)):
    setattr(descriptor, 'column' + str(i),
            df[col].values.ctypes.data_as(ctypes.POINTER(dtype)))
  return descriptor
