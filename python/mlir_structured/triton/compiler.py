import ctypes
import os

from triton.compiler.code_generator import ast_to_ttir

from mlir_structured.execution_engine import ExecutionEngine
from mlir_structured.ir import Context, Module, StringAttr, SymbolTable
from mlir_structured.passmanager import PassManager
from mlir_structured.dialects import triton as tt

__all__ = [
    "compile",
]

_MLIR_ASYNC_RUNTIME_LIB_ENV = "MLIR_ASYNC_RUNTIME_LIB"
_MLIR_ASYNC_RUNTIME_LIB_DEFAULT = "libmlir_async_runtime.so"
_MLIR_C_RUNNER_UTILS_LIB_ENV = "MLIR_C_RUNNER_UTILS_LIB"
_MLIR_C_RUNNER_UTILS_LIB_DEFAULT = "libmlir_c_runner_utils.so"
_MLIR_RUNNER_UTILS_LIB_ENV = "MLIR_RUNNER_UTILS_LIB"
_MLIR_RUNNER_UTILS_LIB_DEFAULT = "libmlir_runner_utils.so"


class CompiledKernel:
  '''This class serves as a drop-in replacement for
     `triton.compiler.compiler.CompiledKernel` for usage in `jit.py`.'''

  def __init__(self, context, module, engine, signature):
    self.context = context
    self.module = module
    self.engine = engine
    self.signature = signature

    # Add dummy properties that are accessed in jit.py.
    self.num_warps = None
    self.shared = None
    self.cu_function = None

  def c_wrapper(self, *args, **kwargs):
    '''This function serves as a drop-in replacement for the original function
       with the same name for usage in `jit.py`. It converts the kernel
       arguments into the format that MLIR's `ExecutionEngine` understands, and,
       for the time being, emulates grid execution with a nested loop.'''

    # Extract useful arguments.
    n_x, n_y, n_z = args[0:3]
    kernel_args = args[10:]

    # Extract kernel arguments.
    ctype_args = []
    for value, (idx, ty) in zip(kernel_args, self.signature.items()):
      if ty[0] == '*':
        addr = value.data_ptr()
        ctype_arg = ctypes.cast(addr, ctypes.c_void_p)
      else:
        to_ctype_value = {
            "i1": ctypes.c_int32,
            "i8": ctypes.c_int8,
            "i16": ctypes.c_int16,
            "i32": ctypes.c_int32,
            "i64": ctypes.c_int64,
            "u32": ctypes.c_uint32,
            "u64": ctypes.c_uint64,
            "fp16": ctypes.c_float,
            "bf16": ctypes.c_float,
            "fp32": ctypes.c_float,
            "f32": ctypes.c_float,
            "fp64": ctypes.c_double,
        }[ty]
        ctype_arg = to_ctype_value(value)
      ctype_arg = ctypes.pointer(ctype_arg)
      ctype_args.append(ctype_arg)

    # Run kernel through the grid wrapper.
    N_args = [ctypes.pointer(ctypes.c_int32(n)) for n in [n_x, n_y, n_z]]
    self.engine.invoke('kernel_grid', *(N_args + ctype_args))


def compile(fn, **kwargs):
  '''This function serves as drop-in replacement for `triton.compile` but uses
     a custom pipeline using MLIR Python bindings and returns a custom
     `CompiledKernel`.'''

  # Convert AST to Triton IR.
  configs = kwargs['configs']
  constants = kwargs['constants']
  debug = kwargs['debug']
  signature = kwargs['signature']
  ttir = ast_to_ttir(fn,
                     signature,
                     configs[0],
                     constants,
                     debug=debug,
                     arch=None)

  with Context() as ctx:
    # Parse Triton IR in our extension.
    tt.register_dialect()
    try:
      mod = Module.parse(str(ttir))
    except Exception as e:
      raise RuntimeError(f'Failed to parse Triton IR:\n\n{ttir}') from e

    # Find function name of the kernel: there should only be one public function
    # at this point.
    kernel_func_name = None
    for op in mod.body.operations:
      if 'sym_name' in op.attributes and 'sym_visibility' in op.attributes:
        visibility = StringAttr(op.attributes['sym_visibility']).value
        if visibility != 'public':
          continue
        assert not kernel_func_name
        kernel_func_name = StringAttr(op.attributes['sym_name']).value
    assert kernel_func_name

    # Replace kernel function name a fixed name such that we can call it easily.
    symbol_table = SymbolTable(mod.operation)
    src_sym = symbol_table[kernel_func_name]
    dst_sym = 'kernel'
    SymbolTable.set_symbol_name(src_sym, dst_sym)
    SymbolTable.replace_all_symbol_uses(kernel_func_name, dst_sym,
                                        mod.operation)

    # Compile with custom pipeline.
    pm = PassManager.parse('builtin.module('
                           '  convert-triton-func-to-func,'
                           '  convert-triton-spmd-to-func-args,'
                           '  func.func(llvm-request-c-wrappers),'
                           '  convert-triton-to-llvm,'
                           '  async-parallel-for,'
                           '  async-to-async-runtime,'
                           '  async-runtime-ref-counting,'
                           '  async-runtime-ref-counting-opt,'
                           '  convert-elementwise-to-linalg,'
                           '  linalg-fuse-elementwise-ops,'
                           '  empty-tensor-to-alloc-tensor,'
                           '  inline,'
                           '  one-shot-bufferize,'
                           '  func.func(convert-linalg-to-loops),'
                           '  convert-async-to-llvm,'
                           '  convert-scf-to-cf,'
                           '  finalize-memref-to-llvm,'
                           '  arith-expand,'
                           '  memref-expand,'
                           '  convert-func-to-llvm,'
                           '  canonicalize'
                           ')')
    try:
      pm.run(mod.operation)
    except Exception as e:
      raise RuntimeError(f'Failed compile Triton IR:\n\n{ttir}') from e

    # Create execution engine.
    try:
      shared_libs = [
          os.getenv(_MLIR_RUNNER_UTILS_LIB_ENV, _MLIR_RUNNER_UTILS_LIB_DEFAULT),
          os.getenv(_MLIR_C_RUNNER_UTILS_LIB_ENV,
                    _MLIR_C_RUNNER_UTILS_LIB_DEFAULT),
          os.getenv(_MLIR_ASYNC_RUNTIME_LIB_ENV,
                    _MLIR_ASYNC_RUNTIME_LIB_DEFAULT)
      ]

      engine = ExecutionEngine(mod, shared_libs=shared_libs, opt_level=3)
    except Exception as e:
      raise RuntimeError(f'Failed to create execution engine.\n\n'
                         f'Triton IR:\n\n{ttir}\n\n'
                         f'Compiled IR:\n\n{mod}\n\n') from e

    return CompiledKernel(ctx, mod, engine, signature)
