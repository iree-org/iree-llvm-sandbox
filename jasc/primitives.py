# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Jax primitives backing Jasc schedules."""

from collections.abc import Callable, Sequence
import contextlib
import itertools
from typing import Any, Optional

import jax
from jax.extend.linear_util import wrap_init
from jax.interpreters import mlir as jax_mlir
from jax.interpreters import partial_eval as pe
from jax.lib import xla_client
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import pdl
from jaxlib.mlir.dialects import stablehlo
from jaxlib.mlir.dialects import transform

import call_kernel
from dialect import jasc as jasc_dialect


_JAX_COMPATIBLE_LOWERING = True

call_kernel.init_llvm()


@contextlib.contextmanager
def enable_jasc_lowering():
  """ContextManager to enable usage of `with enable_jasc_lowering()`."""
  global _JAX_COMPATIBLE_LOWERING
  _JAX_COMPATIBLE_LOWERING = False
  try:
    yield
  finally:
    _JAX_COMPATIBLE_LOWERING = True


def _func_to_mlir_module(
    ctx: jax_mlir.LoweringRuleContext, func: Callable[..., Any]
) -> ir.Module:
  """Compiles func to an MLIR module."""
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrap_init(func), ctx.avals_in)
  closed_jaxpr = jax.core.ClosedJaxpr(jaxpr, consts)
  result = jax_mlir.lower_jaxpr_to_module(
      module_name='jasc_jit',
      jaxpr=closed_jaxpr,
      backend_or_name=ctx.module_context.backend_or_name,
      ordered_effects=[],
      name_stack=ctx.module_context.name_stack,
      donated_args=[False] * len(closed_jaxpr.jaxpr.invars),
      axis_context=ctx.module_context.axis_context,
      platforms=ctx.module_context.platforms,
      lowering_parameters=jax_mlir.LoweringParameters(),
  )
  if result.keepalive or result.host_callbacks:
    raise NotImplementedError('Jasc does not support callbacks')
  return result.module


def _generate_schedule(build_schedule: Callable[[ir.Value], None]) -> None:
  sequence_op = transform.SequenceOp(
      transform.FailurePropagationMode.Propagate, (), pdl.OperationType.get()
  )
  with ir.InsertionPoint(sequence_op.body):
    build_schedule(sequence_op.bodyTarget)
    transform.YieldOp()


def _jit_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    *args: ir.Value,
    func: Callable[..., Any],
    module: Optional[ir.Module] = None,
    build_schedule: Callable[[ir.Value], None],
    out_avals: Sequence[jax.core.AbstractValue],
    dump_ir: bool,
) -> Sequence[ir.Value]:
  """Lowers a call to the jit primitive.

  Args:
    ctx: Jax lowering context.
    *args: MLIR values that holds the value of the flattened function arguments.
    func: Function to lower.
    module: Optional already lowered representation of func. If this is supplied
      it will be used rather than lowering `func`.
    build_schedule: Function that generates an MLIR transform dialect script.
      Takes the root transform handle as input and expects the insertion point
      to be set.
    out_avals: Abstract values of func outputs.
    dump_ir: If true, log intermediate steps of the compilation process.

  Returns:
    A sequence of MLIR values holding the value of the function outputs.
  """
  del out_avals
  if module is None:
    with enable_jasc_lowering():
      lowered_ir = _func_to_mlir_module(ctx, func)
  else:
    lowered_ir = module
  with lowered_ir.context:
    with ir.Location.unknown(lowered_ir.context):
      with ir.InsertionPoint(lowered_ir.body):
        _generate_schedule(build_schedule)

  backend_config = None
  mlir_args = []

  if len(ctx.module_context.platforms) > 1:
    raise NotImplementedError('Multi-platform lowering')
  if ctx.module_context.platforms[0] == 'cpu':
    compiled_kernel = call_kernel.create_cpu_kernel(
        module=lowered_ir,
        num_inputs=len(args),
        num_outputs=len(ctx.avals_out),
        dump_ir=dump_ir,
    )
    ctx.module_context.add_keepalive(compiled_kernel)
    identifier_attr = jax_mlir.dense_int_elements([compiled_kernel.identifier])
    identifier_op = stablehlo.ConstantOp(identifier_attr)
    mlir_args = [identifier_op.result]
  else:
    raise NotImplementedError(
        f'Jasc does not support platform {ctx.module_context.platforms[0]}'
    )

  mlir_args.extend(args)
  out_types = tuple(
      itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out))
  )
  custom_call = stablehlo.CustomCallOp(
      out_types,
      mlir_args,
      call_target_name='jasc.call_kernel',
      backend_config=backend_config,
  )
  return custom_call.results


jit_p = jax.core.Primitive('jasc.jit')
jit_p.multiple_results = True
jit_p.def_impl(
    lambda *args, func, module, build_schedule, out_avals, dump_ir: func(*args)
)
jit_p.def_abstract_eval(
    lambda *args, func, module, build_schedule, out_avals, dump_ir: out_avals
)
jax_mlir.register_lowering(jit_p, _jit_lowering)


xla_client.register_custom_call_target(
    'jasc.call_kernel', call_kernel.get_cpu_callback(), platform='cpu'
)


def _tag_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    *args: ir.Value,
    func: Callable[..., Any],
    out_avals: Sequence[jax.core.AbstractValue],
    name: str,
) -> Sequence[ir.Value]:
  """Lowers a call to the tag primitive.

  Args:
    ctx: Jax lowering context.
    *args: MLIR values that holds the value of the flattened function arguments.
    func: Function to tag.
    out_avals: Abstract values of func outputs.
    name: Tag name.

  Returns:
    A sequence of MLIR values holding the value of the function outputs.
  """
  del out_avals
  if _JAX_COMPATIBLE_LOWERING:
    return jax_mlir.lower_fun(func, multiple_results=True)(ctx, *args)

  jasc_dialect.register_and_load_dialect(ctx.module_context.context)
  out_types = tuple(
      itertools.chain(*map(jax_mlir.aval_to_ir_types, ctx.avals_out))
  )
  tag_op = jasc_dialect.TagRegionOp(out_types, name)
  tag_op.body.blocks.append()
  with ir.InsertionPoint(tag_op.body.blocks[0]):
    lower_rule = jax_mlir.lower_fun(func, multiple_results=True)
    results = lower_rule(ctx, *args)
    jasc_dialect.ReturnOp(sum(results, ()))
  return tag_op.results


tag_p = jax.core.Primitive('jasc.tag')
tag_p.multiple_results = True
tag_p.def_impl(lambda *args, func, out_avals, name: func(*args))
tag_p.def_abstract_eval(lambda *args, func, out_avals, name: out_avals)
jax_mlir.register_lowering(tag_p, _tag_lowering)
