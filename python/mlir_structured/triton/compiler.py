import ctypes

from triton.compiler.code_generator import ast_to_ttir

from mlir_structured.execution_engine import ExecutionEngine
from mlir_structured.ir import Context, Module, StringAttr, SymbolTable
from mlir_structured.passmanager import PassManager
from mlir_structured.dialects import triton as tt

__all__ = [
    "compile",
]


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

    # Run kernel in manually in a grid.
    # TODO(ingomueller): This is a *very* simplistic way to simulate the grid!
    #     It allows us to bootstrap the compiler and invocation infrastructure
    #     but will soon need to be replaced with some proper SPMD concept.
    for x in range(n_x):
      for y in range(n_y):
        for z in range(n_z):
          self.engine.invoke('kernel', *ctype_args)


def compile(fn, **kwargs):
  '''This function serves as drop-in replacement for `triton.compile` but uses
     a custom pipeline using MLIR Python bindings and returns a custom
     `CompiledKernel`.'''

  # Convert AST to Triton IR.
  configs = kwargs['configs']
  constants = kwargs['constants']
  debug = kwargs['debug']
  signature = kwargs['signature']
  ttir = ast_to_ttir(fn, signature, configs[0], constants, debug=debug)

  with Context() as ctx:
    # Parse Triton IR in our extension.
    tt.register_dialect()
    try:
      mod = Module.parse(str(ttir))
    except Exception as e:
      raise RuntimeError(f'Failed to parse Triton IR:\n\n{ttir}') from e

    # Compile with custom pipeline.
    pm = PassManager.parse('builtin.module('
                           '  convert-triton-to-llvm,'
                           '  convert-arith-to-llvm'
                           ')')
    try:
      pm.run(mod.operation)
    except Exception as e:
      raise RuntimeError(f'Failed compile Triton IR:\n\n{ttir}') from e

    # Find exact function name of kernel.
    kernel_name = None
    for op in mod.body.operations:
      if 'sym_name' in op.attributes:
        sym_name = StringAttr(op.attributes['sym_name']).value
        if sym_name.startswith('kernel'):
          assert kernel_name is None
          kernel_name = sym_name
    assert kernel_name

    # Replace kernel function name with name implementing C interface and
    # without variant-specific suffix such that we can call it from the
    # ExecutionEngine using a fixed name.
    symbol_table = SymbolTable(mod.operation)
    src_sym = symbol_table[kernel_name]
    dst_sym = "_mlir_ciface_kernel"
    SymbolTable.set_symbol_name(src_sym, dst_sym)
    SymbolTable.replace_all_symbol_uses(kernel_name, dst_sym, mod.operation)

    # Create execution engine.
    try:
      engine = ExecutionEngine(mod)
    except Exception as e:
      raise RuntimeError(f'Failed to create execution engine.\n\n'
                         f'Triton IR:\n\n{ttir}\n\n'
                         f'Compiled IR:\n\n{mod}\n\n') from e

    return CompiledKernel(ctx, mod, engine, signature)
