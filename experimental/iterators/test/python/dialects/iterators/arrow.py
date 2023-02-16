# RUN: %PYTHON %s | FileCheck %s

import ctypes
import io
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.cffi
import pyarrow.parquet

from mlir_iterators.dialects import iterators as it
from mlir_iterators.dialects import tabular as tab
from mlir_iterators.dialects import tuple as tup
from mlir_iterators.passmanager import PassManager
from mlir_iterators.execution_engine import ExecutionEngine
from mlir_iterators.ir import Context, Module


def run(f):
  print("\nTEST:", f.__name__)
  with Context():
    it.register_dialect()
    tab.register_dialect()
    tup.register_dialect()
    f()
  return f


# MLIR definitions of the C structs of the Arrow ABI.
ARROW_STRUCT_DEFINITIONS_MLIR = '''
    !arrow_schema = !llvm.struct<"ArrowSchema", (
      ptr<i8>,  // format
      ptr<i8>,  // name
      ptr<i8>,  // metadata
      i64,      // flags
      i64,      // n_children
      ptr<ptr<struct<"ArrowSchema">>>,              // children
      ptr<struct<"ArrowSchema">>,                   // dictionary
      ptr<func<void (ptr<struct<"ArrowSchema">>)>>, // release
      ptr       // private_data
      )>
    !arrow_array = !llvm.struct<"ArrowArray", (
      i64,        // length
      i64,        // null_count
      i64,        // offset
      i64,        // n_buffers
      i64,        // n_children
      ptr<ptr>,   // buffers
      ptr<ptr<struct<"ArrowArray">>>,               // children
      ptr<struct<"ArrowArray">>,                    // dictionary
      ptr<func<void (ptr<struct<"ArrowArray">>)>>,  // release
      ptr         // private_data
      )>
    !arrow_array_stream = !llvm.struct<"ArrowArrayStream", (
      ptr<func<i32 (ptr<struct<"ArrowArrayStream">>, ptr<!arrow_schema>)>>, // get_schema
      ptr<func<i32 (ptr<struct<"ArrowArrayStream">>, ptr<!arrow_array>)>>,  // get_next
      ptr<func<ptr<i8> (ptr<struct<"ArrowArrayStream">>)>>, // get_last_error
      ptr<func<void (ptr<struct<"ArrowArrayStream">>)>>,    // release
      ptr   // private_data
      )>
    '''

# Arrow data types that are currently supported.
ARROW_SUPPORTED_TYPES = [
    pa.int8(),
    pa.int16(),
    pa.int32(),
    pa.int64(),
    pa.float16(),
    pa.float32(),
    pa.float64()
]


# Converts the given Arrow type to the name of the corresponding MLIR type.
def to_mlir_type(t: pa.DataType) -> str:
  if pa.types.is_signed_integer(t):
    return 'i' + str(t.bit_width)
  if pa.types.is_floating(t):
    return 'f' + str(t.bit_width)
  raise NotImplementedError("Only floats and signed integers supported")


# Compiles the given code and wraps it into an execution engine.
def build_and_create_engine(code: str) -> ExecutionEngine:
  mod = Module.parse(ARROW_STRUCT_DEFINITIONS_MLIR + code)
  pm = PassManager.parse('builtin.module('
                         'convert-iterators-to-llvm,'
                         'convert-tabular-to-llvm,'
                         'decompose-iterator-states,'
                         'decompose-tuples,'
                         'one-shot-bufferize,'
                         'canonicalize,cse,'
                         'expand-strided-metadata,'
                         'finalize-memref-to-llvm,'
                         'canonicalize,cse,'
                         'convert-func-to-llvm,'
                         'reconcile-unrealized-casts,'
                         'convert-scf-to-cf,'
                         'convert-cf-to-llvm)')
  pm.run(mod.operation)
  runtime_lib = os.environ['ITERATORS_RUNTIME_LIBRARY_PATH']
  engine = ExecutionEngine(mod, shared_libs=[runtime_lib])
  return engine


# Generate MLIR that reads the arrays of an Arrow array stream and produces (and
# prints) the element-wise sum of each array.
def generate_sum_batches_elementwise_code(schema: pa.Schema) -> str:
  mlir_types = [to_mlir_type(t) for t in schema.types]

  # Generate code that, for each type, extracts rhs and lhs struct values, adds
  # them, and then inserts the result into a result struct.
  elementwise_sum = f'''
    %lhsvals:{len(mlir_types)} = tuple.to_elements %lhs : !tuple_type
    %rhsvals:{len(mlir_types)} = tuple.to_elements %rhs : !tuple_type
  '''
  for i, t in enumerate(mlir_types):
    elementwise_sum += f'''
        %sum{i} = arith.add{t[0]} %lhsvals#{i}, %rhsvals#{i} : {t}
        '''
  result_vars = ', '.join((f'%sum{i}' for i in range(len(mlir_types))))
  elementwise_sum += f'''
    %result = tuple.from_elements {result_vars} : !tuple_type
  '''

  # Adapt main program to types of the given schema.
  code = f'''
    !tuple_type = tuple<{', '.join(mlir_types)}>
    !tabular_view_type = !tabular.tabular_view<{', '.join(mlir_types)}>

    // Add numbers of two structs element-wise.
    func.func private @sum_struct(%lhs : !tuple_type, %rhs : !tuple_type) -> !tuple_type {{
      {elementwise_sum}
      return %result : !tuple_type
    }}

    // Consume the given tabular view and produce one element-wise sum from the elements.
    func.func @sum_tabular_view(%tabular_view: !tabular_view_type) -> !tuple_type  {{
      %tabular_stream = iterators.tabular_view_to_stream %tabular_view
        to !iterators.stream<!tuple_type>
      %reduced = "iterators.reduce"(%tabular_stream) {{reduceFuncRef = @sum_struct}}
        : (!iterators.stream<!tuple_type>) -> (!iterators.stream<!tuple_type>)
      %result:2 = iterators.stream_to_value %reduced : !iterators.stream<!tuple_type>
      return %result#0 : !tuple_type
    }}

    // For each Arrow array in the input stream, produce an element-wise sum.
    func.func @main(%arrow_stream: !llvm.ptr<!arrow_array_stream>)
        attributes {{ llvm.emit_c_interface }} {{
      %tabular_view_stream = iterators.from_arrow_array_stream %arrow_stream
          to !iterators.stream<!tabular_view_type>
      %sums = "iterators.map"(%tabular_view_stream) {{mapFuncRef = @sum_tabular_view}}
        : (!iterators.stream<!tabular_view_type>) -> (!iterators.stream<!tuple_type>)
      "iterators.sink"(%sums) : (!iterators.stream<!tuple_type>) -> ()
      return
    }}
  '''

  return code


# Feeds the given Arrow array stream/record batch reader into an MLIR kernel
# that reads the arrays the stream and produces (and prints) the element-wise
# sum of each array/record batch.
def sum_batches_elementwise_with_iterators(
    record_batch_reader: pa.RecordBatchReader) -> None:

  code = generate_sum_batches_elementwise_code(record_batch_reader.schema)
  engine = build_and_create_engine(code)

  # Create C struct describing the record batch reader.
  ffi = pa.cffi.ffi
  cffi_stream = ffi.new('struct ArrowArrayStream *')
  cffi_stream_ptr = int(ffi.cast("intptr_t", cffi_stream))
  record_batch_reader._export_to_c(cffi_stream_ptr)

  # Wrap argument and invoke compiled function.
  arg = ctypes.pointer(ctypes.cast(cffi_stream_ptr, ctypes.c_void_p))
  engine.invoke('main', arg)


# Create a sample Arrow table with one column per supported type.
def create_test_input() -> pa.Table:
  # Use pyarrow to create an Arrow table in memory.
  fields = [pa.field(str(t), t, False) for t in ARROW_SUPPORTED_TYPES]
  schema = pa.schema(fields)
  arrays = [
      pa.array(np.array(np.arange(10) + 100 * i, field.type.to_pandas_dtype()))
      for i, field in enumerate(fields)
  ]
  table = pa.table(arrays, schema)
  return table


# Test case: Read from a sequence of Arrow arrays/record batches (produced by a
# Python generator).


# CHECK-LABEL: TEST: testArrowStreamInput
# CHECK-NEXT: (10, 510, 1010, 1510, 2010, 2510, 3010)
# CHECK-NEXT: (35, 535, 1035, 1535, 2035, 2535, 3035)
@run
def testArrowStreamInput():
  # Use pyarrow to create an Arrow table in memory.
  table = create_test_input()

  # Make physically separate batches from the table. (This ensures offset=0).
  batches = (b for batch in table.to_batches(max_chunksize=5)
             for b in pa.Table.from_pandas(batch.to_pandas()).to_batches())

  # Create a RecordBatchReader and export it as a C struct.
  reader = pa.RecordBatchReader.from_batches(table.schema, batches)

  # Hand the reader as an Arrow array stream to the Iterators test program.
  sum_batches_elementwise_with_iterators(reader)


# Test case: Read data from a Parquet file (through pyarrow's C++-implemented
# Parquet reader).


# CHECK-LABEL: TEST: testArrowParquetInput
# CHECK-NEXT: (10, 510, 1010, 1510, 2510, 3010)
# CHECK-NEXT: (35, 535, 1035, 1535, 2535, 3035)
@run
def testArrowParquetInput():
  table = create_test_input()
  # Remove f16 column, which the Parquet reader/writer doesn't support yet.
  table = table.drop(['halffloat'])

  # Create a tempororay in-memory file with test data.
  with io.BytesIO() as temp_file, \
      pa.PythonFile(temp_file) as parquet_file:
    # Export test data as Parquet.
    pa.parquet.write_table(table, parquet_file)

    # Flush and rewind to the beginning of the file.
    parquet_file.flush()
    temp_file.flush()
    temp_file.seek(0)

    # Open as ParquetFile instance.
    parquet_file = pa.parquet.ParquetFile(temp_file)

    # Create a Python generator of batches (which reads record batches using the
    # C++ implementation) and turn that into a RecordBatchReader.
    # TODO: It may be possible to get a RecordBatchReader for the Parquet file
    # directly (i.e., without going through a Python generator) but I did not
    # see it exposed to Python.
    batches_generator = parquet_file.iter_batches(batch_size=5)
    reader = pa.RecordBatchReader.from_batches(table.schema, batches_generator)

    # Hand the reader as an Arrow array stream to the Iterators test program.
    sum_batches_elementwise_with_iterators(reader)


# Test case: Read from a sequence of Arrow arrays/record batches (produced by a
# Python generator).


# Create a generator that produces single-row record batches with increasing
# numbers with an artificial delay of one second after each of them. Since each
# generated record batch immediately produces output, this visually demonstrate
# that the consumption by the MLIR-based iterators interleaves with the
# Python-based production of the record batches in the stream.
def generate_batches_with_delay(schema: pa.Schema) -> None:
  for i in range(5):
    arrays = [
        pa.array(np.array([i], field.type.to_pandas_dtype()))
        for field in schema
    ]
    batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
    yield batch
    # Sleep only when a TTY is attached (in order not to delay unit tests).
    if sys.stdout.isatty():
      time.sleep(1)


# CHECK-LABEL: TEST: testGeneratorInput
# CHECK-NEXT: (0, 0, 0, 0, 0, 0, 0)
# CHECK-NEXT: (1, 1, 1, 1, 1, 1, 1)
# CHECK-NEXT: (2, 2, 2, 2, 2, 2, 2)
# CHECK-NEXT: (3, 3, 3, 3, 3, 3, 3)
# CHECK-NEXT: (4, 4, 4, 4, 4, 4, 4)
@run
def testGeneratorInput():
  # Use pyarrow to create an Arrow table in memory.
  table = create_test_input()

  # Make physically separate batches from the table. (This ensures offset=0).
  generator = generate_batches_with_delay(table.schema)

  # Create a RecordBatchReader and export it as a C struct.
  reader = pa.RecordBatchReader.from_batches(table.schema, generator)

  # Hand the reader as an Arrow array stream to the Iterators test program.
  sum_batches_elementwise_with_iterators(reader)
