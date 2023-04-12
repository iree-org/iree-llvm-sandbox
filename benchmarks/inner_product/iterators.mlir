// This file is a "template" implementation of the dot product (inner product)
// used by the benchmark `dot_product.py`. It is designed to be (1) valid MLIR
// and (2) simple to modify with string manipulation to change the type. By
// changing the aliases `!element_type`, `!int_type`, and `!float_type` as well
// as the attributes `mapFuncRef = @mul_struct_int` and
// `reduceFuncRef = @sum_int`, it is possible to change the element type of the
// implementation to any built-in integer or float type.

// Type aliases for element types. Change these to "instantiate" the template.
!element_type = i32
!int_type = i32
!float_type = f32

// Derived types.
!struct_type = !llvm.struct<(!element_type,!element_type)>
!struct_type_int = !llvm.struct<(!int_type,!int_type)>
!struct_type_float = !llvm.struct<(!float_type,!float_type)>

//
// Implementation for integer types.
//
func.func private @mul_struct_int(%struct : !struct_type_int) -> !int_type {
  %lhs = llvm.extractvalue %struct[0] : !struct_type_int
  %rhs = llvm.extractvalue %struct[1] : !struct_type_int
  %sum = arith.muli %lhs, %rhs : !int_type
  return %sum : !int_type
}

func.func private @sum_int(%lhs : !int_type, %rhs : !int_type) -> !int_type {
  %result = arith.addi %lhs, %rhs : !int_type
  return %result : !int_type
}

//
// Implementation for float types.
//
func.func private @mul_struct_float(%struct : !struct_type_float) -> !float_type {
  %lhs = llvm.extractvalue %struct[0] : !struct_type_float
  %rhs = llvm.extractvalue %struct[1] : !struct_type_float
  %sum = arith.mulf %lhs, %rhs : !float_type
  return %sum : !float_type
}

func.func private @sum_float(%lhs : !float_type, %rhs : !float_type) -> !float_type {
  %result = arith.addf %lhs, %rhs : !float_type
  return %result : !float_type
}

//
// Main program.
//
func.func @main(%input: !tabular.tabular_view<!element_type,!element_type>,
                %output: !llvm.ptr<!element_type>)
    attributes { llvm.emit_c_interface } {
  %stream = iterators.tabular_view_to_stream %input
    to !iterators.stream<!struct_type>
  %summed = "iterators.map"(%stream) {mapFuncRef = @mul_struct_int}
    : (!iterators.stream<!struct_type>) -> (!iterators.stream<!element_type>)
  %reduced = "iterators.reduce"(%summed) {reduceFuncRef = @sum_int}
    : (!iterators.stream<!element_type>) -> (!iterators.stream<!element_type>)
  %result:2 = iterators.stream_to_value %reduced : !iterators.stream<!element_type>
  llvm.store %result#0, %output : !llvm.ptr<!element_type>
  return
}
