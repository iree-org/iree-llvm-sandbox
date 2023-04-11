# RUN: %PYTHON %s | FileCheck %s

import os

import numpy as np
import pyarrow as pa
import pyarrow.cffi

# Define type short-hands.
i8 = pa.int8()
u8 = pa.uint8()
i16 = pa.int16()
u16 = pa.uint16()
i32 = pa.int32()
u32 = pa.uint32()
i64 = pa.int64()
u64 = pa.uint64()
f16 = pa.float16()
f32 = pa.float32()
f64 = pa.float64()

#===------------------------------------------------------------------------===#
# Load runtime lib as CFFI lib.
#===------------------------------------------------------------------------===#
# Reuse ffi object of pyarrow, which has definitions of the C interface types.
ffi = pa.cffi.ffi

# Load our functions.
GET_COLUMN_FUNCS_NAMES = {
    i8: ('signed char', 'mlirIteratorsArrowArrayGetInt8Column'),
    u8: ('unsigned char', 'mlirIteratorsArrowArrayGetUInt8Column'),
    i16: ('signed short', 'mlirIteratorsArrowArrayGetInt16Column'),
    u16: ('unsigned short', 'mlirIteratorsArrowArrayGetUInt16Column'),
    i32: ('signed int', 'mlirIteratorsArrowArrayGetInt32Column'),
    u32: ('unsigned int', 'mlirIteratorsArrowArrayGetUInt32Column'),
    i64: ('signed long', 'mlirIteratorsArrowArrayGetInt64Column'),
    u64: ('unsigned long', 'mlirIteratorsArrowArrayGetUInt64Column'),
    f16: ('unsigned short', 'mlirIteratorsArrowArrayGetFloat16Column'),
    f32: ('float', 'mlirIteratorsArrowArrayGetFloat32Column'),
    f64: ('double', 'mlirIteratorsArrowArrayGetFloat64Column'),
}

for pointee_type, func_name in GET_COLUMN_FUNCS_NAMES.values():
  ffi.cdef('const {} * {}(struct ArrowArray *array,'
           '              struct ArrowSchema *schema,'
           '              long long i);'.format(pointee_type, func_name))

ffi.cdef('''
    long long mlirIteratorsArrowArrayGetSize(struct ArrowArray *array);
    void mlirIteratorsArrowArrayRelease(struct ArrowArray *array);
    void mlirIteratorsArrowSchemaRelease(struct ArrowSchema *schema);
    bool mlirIteratorsArrowArrayStreamGetNext(struct ArrowArrayStream *stream,
                                              struct ArrowArray *result);
    void mlirIteratorsArrowArrayStreamGetSchema(struct ArrowArrayStream *stream,
                                                struct ArrowSchema *result);
    void mlirIteratorsArrowArrayStreamRelease(struct ArrowArrayStream *stream);
    ''')

# Dlopen our library.
runtime_lib_path = os.environ['ITERATORS_RUNTIME_LIBRARY_PATH']
lib = ffi.dlopen(runtime_lib_path)

#===------------------------------------------------------------------------===#
# Set up test data.
#===------------------------------------------------------------------------===#
# Define schema and table.
fields = [pa.field(str(t), t, False) for t in GET_COLUMN_FUNCS_NAMES.keys()]
schema = pa.schema(fields)

# CHECK-LABEL: schema:
# CHECK-NEXT:  int8: int8 not null
# CHECK-NEXT:  uint8: uint8 not null
# CHECK-NEXT:  int16: int16 not null
# CHECK-NEXT:  uint16: uint16 not null
# CHECK-NEXT:  int32: int32 not null
# CHECK-NEXT:  uint32: uint32 not null
# CHECK-NEXT:  int64: int64 not null
# CHECK-NEXT:  uint64: uint64 not null
# CHECK-NEXT:  halffloat: halffloat not null
# CHECK-NEXT:  float: float not null
# CHECK-NEXT:  double: double not null

print("schema:")
print(schema)

arrays = [
    pa.array(np.array(np.arange(10) + 100 * i, t.to_pandas_dtype()))
    for i, t in enumerate(GET_COLUMN_FUNCS_NAMES.keys())
]
table = pa.table(arrays, schema)
batch = table.to_batches()[0]

# CHECK-LABEL: original batch:
# CHECK-NEXT:  [0 1 2 3 4 5 6 7 8 9]
# CHECK-NEXT:  [100 101 102 103 104 105 106 107 108 109]
# CHECK-NEXT:  [200 201 202 203 204 205 206 207 208 209]
# CHECK-NEXT:  [300 301 302 303 304 305 306 307 308 309]
# CHECK-NEXT:  [400 401 402 403 404 405 406 407 408 409]
# CHECK-NEXT:  [500 501 502 503 504 505 506 507 508 509]
# CHECK-NEXT:  [600 601 602 603 604 605 606 607 608 609]
# CHECK-NEXT:  [700 701 702 703 704 705 706 707 708 709]
# CHECK-NEXT:  [800. 801. 802. 803. 804. 805. 806. 807. 808. 809.]
# CHECK-NEXT:  [900. 901. 902. 903. 904. 905. 906. 907. 908. 909.]
# CHECK-NEXT:  [1000. 1001. 1002. 1003. 1004. 1005. 1006. 1007. 1008. 1009.]

print("original batch:")
for c in batch.columns:
  print(c.to_numpy())

#===------------------------------------------------------------------------===#
# Test C data interface.
#===------------------------------------------------------------------------===#
# Create C struct describing the batch.
cffi_batch = ffi.new('struct ArrowArray *')
batch._export_to_c(int(ffi.cast("intptr_t", cffi_batch)))

# Create C struct describing schema.
cffi_schema = ffi.new('struct ArrowSchema *')
schema._export_to_c(int(ffi.cast("intptr_t", cffi_schema)))

# Test function returning batch size.
batch_size = lib.mlirIteratorsArrowArrayGetSize(cffi_batch)

# CHECK-LABEL: retrieved batch size:
# CHECK-NEXT:  10

print("retrieved batch size:")
print(batch_size)

# CHECK-LABEL: retrieved batch:
# CHECK-NEXT:  [0 1 2 3 4 5 6 7 8 9]
# CHECK-NEXT:  [100 101 102 103 104 105 106 107 108 109]
# CHECK-NEXT:  [200 201 202 203 204 205 206 207 208 209]
# CHECK-NEXT:  [300 301 302 303 304 305 306 307 308 309]
# CHECK-NEXT:  [400 401 402 403 404 405 406 407 408 409]
# CHECK-NEXT:  [500 501 502 503 504 505 506 507 508 509]
# CHECK-NEXT:  [600 601 602 603 604 605 606 607 608 609]
# CHECK-NEXT:  [700 701 702 703 704 705 706 707 708 709]
# CHECK-NEXT:  [800. 801. 802. 803. 804. 805. 806. 807. 808. 809.]
# CHECK-NEXT:  [900. 901. 902. 903. 904. 905. 906. 907. 908. 909.]
# CHECK-NEXT:  [1000. 1001. 1002. 1003. 1004. 1005. 1006. 1007. 1008. 1009.]

# Test functions accessing columns from the batch.
print("retrieved batch:")
for i, (type, (_, func_name)) in enumerate(GET_COLUMN_FUNCS_NAMES.items()):
  func = lib.__getattr__(func_name)

  # Call function, which returns a pointer.
  ptr = func(cffi_batch, cffi_schema, i)

  # Wrap the pointer into a buffer, convert that into a type numpy array.
  buffer = ffi.buffer(ptr, batch_size * type.bit_width // 8)
  array = np.frombuffer(buffer, dtype=type.to_pandas_dtype())

  print(array)

# Release memory owned by the C structs.
lib.mlirIteratorsArrowArrayRelease(cffi_batch)
lib.mlirIteratorsArrowSchemaRelease(cffi_schema)

#===------------------------------------------------------------------------===#
# Test C stream interface.
#===------------------------------------------------------------------------===#
reader = pa.RecordBatchReader.from_batches(schema,
                                           table.to_batches(max_chunksize=5))

# Create C struct describing record batch reader.
cffi_stream = ffi.new('struct ArrowArrayStream *')
reader._export_to_c(int(ffi.cast("intptr_t", cffi_stream)))

# Get schema and import it into pyarrow.
lib.mlirIteratorsArrowArrayStreamGetSchema(cffi_stream, cffi_schema)
schema = pa.Schema._import_from_c(int(ffi.cast("intptr_t", cffi_schema)))

# CHECK-LABEL: schema from stream:
# CHECK-NEXT:  int8: int8 not null
# CHECK-NEXT:  uint8: uint8 not null
# CHECK-NEXT:  int16: int16 not null
# CHECK-NEXT:  uint16: uint16 not null
# CHECK-NEXT:  int32: int32 not null
# CHECK-NEXT:  uint32: uint32 not null
# CHECK-NEXT:  int64: int64 not null
# CHECK-NEXT:  uint64: uint64 not null
# CHECK-NEXT:  halffloat: halffloat not null
# CHECK-NEXT:  float: float not null
# CHECK-NEXT:  double: double not null

print("schema from stream:")
print(schema)

# CHECK-LABEL: batches:
# CHECK-NEXT:     int8  uint8  int16  uint16  int32  uint32  int64  uint64  halffloat  float  double
# CHECK-NEXT:  0     0    100    200     300    400     500    600     700      800.0  900.0  1000.0
# CHECK-NEXT:  1     1    101    201     301    401     501    601     701      801.0  901.0  1001.0
# CHECK-NEXT:  2     2    102    202     302    402     502    602     702      802.0  902.0  1002.0
# CHECK-NEXT:  3     3    103    203     303    403     503    603     703      803.0  903.0  1003.0
# CHECK-NEXT:  4     4    104    204     304    404     504    604     704      804.0  904.0  1004.0
# CHECK-NEXT:     int8  uint8  int16  uint16  int32  uint32  int64  uint64  halffloat  float  double
# CHECK-NEXT:  0     5    105    205     305    405     505    605     705      805.0  905.0  1005.0
# CHECK-NEXT:  1     6    106    206     306    406     506    606     706      806.0  906.0  1006.0
# CHECK-NEXT:  2     7    107    207     307    407     507    607     707      807.0  907.0  1007.0
# CHECK-NEXT:  3     8    108    208     308    408     508    608     708      808.0  908.0  1008.0
# CHECK-NEXT:  4     9    109    209     309    409     509    609     709      809.0  909.0  1009.0

# Iterate over batches provided by stream and print.
print("batches:")
while lib.mlirIteratorsArrowArrayStreamGetNext(cffi_stream, cffi_batch):
  batch = pa.RecordBatch._import_from_c(int(ffi.cast("intptr_t", cffi_batch)),
                                        schema)
  print(batch.to_pandas().to_string())

# Release memory owned by the C structs.
lib.mlirIteratorsArrowArrayRelease(cffi_batch)
lib.mlirIteratorsArrowSchemaRelease(cffi_schema)
lib.mlirIteratorsArrowArrayStreamRelease(cffi_stream)
