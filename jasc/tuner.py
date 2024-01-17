# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Utilities for custom autotuners for parametric transforms."""

from __future__ import annotations

import abc
from collections.abc import Sequence
import dataclasses
import io
import math
import timeit
from typing import Any, Callable, Optional

import jax
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import transform

from jasc import jasc
from transform_ops import jasc_transform_ops

@dataclasses.dataclass
class TunerBase(abc.ABC):
  """Base class for custom autotuners.

  Provides a default autotuning loop based on a budget of tuning configurations
  to evaluate. Subclasses have to implement `get_tuning_config` to drive the
  autotuning process.
  Currently evaluation of a tuned function is limited to the metric of execution
  time.
  """

  func: Callable[[Any], Any]
  parametric_schedule: jasc.Schedule
  inputs: Sequence[jax.Array]
  budget: int = 10
  tuned_func_evals: int = 100
  dump_ir: bool = False

  @abc.abstractmethod
  def get_tuning_config(
      self,
      tuning_vars: Sequence[jasc_transform_ops.TuningParamOp],
      previous_results: Sequence[TuningResult],
  ) -> TuningConfig:
    """Returns a tuning configuration to specialize a parametric schedule.

    A tuning configuration is a list of explicit values for the set of tuning
    variables in a parametric schedule. The type of value is currently limited
    to int.
    Args:
      tuning_vars: The list of tuning variables in a parametric schedule
      previous_results: The previously evaluated configurations and respective
        result metric for this set of tuning variables.
    """
    ...

  def tune(
      self,
  ) -> tuple[
      float, Optional[Callable[..., Any]], Optional[jasc.Schedule], list[float]
  ]:
    """Explores different configurations for the tuning parameters in the schedule."""

    # Lower module to linalg and insert parametric schedule into the IR.
    initial_module = jasc.lower_to_linalg(self.func, *self.inputs)
    jasc.insert_schedule(
        initial_module, self.parametric_schedule, dump_schedule=self.dump_ir
    )

    # Print the initial module to an object so it can be reparsed repeatedly.
    f = io.StringIO("")
    initial_module.operation.print(f)
    tuning_vars = self.get_tuning_vars(initial_module)

    # Tuning loop
    times: list[float] = []
    best_time: float = math.inf
    best_fun: Optional[Callable[..., Any]] = None
    best_schedule: Optional[jasc.Schedule] = None
    previous_configs: list[TuningResult] = []
    for _ in range(self.budget):
      # Create new copy of the module by reparsing it.

      # TODO(mluecke): Reparsing is a slow way of copying but cloning is not yet
      #                exposed in the Python bindings. Implement a better
      #                approach to this.
      module = ir.Module.parse(
          f.getvalue(), context=initial_module.operation.context
      )
      config = self.get_tuning_config(tuning_vars, previous_configs)

      def meta_schedule(module: jasc.OpHandle) -> None:
        sequence_op = module.match_ops(transform.SequenceOp)
        sequence_op.apply_tuning_config(config.values)  # pylint: disable=cell-var-from-loop

      # Apply a tuning configuration to convert the parametric schedule to a
      # version with explicit parameters only.
      jasc.insert_schedule(module, meta_schedule, dump_schedule=self.dump_ir)
      jasc.apply_schedule(module)

      # Evaluate this tuning configuration by applying the schedule the payload
      # IR and timing the execution.
      try:
        tuned_fun = jasc.jit(self.func, module=module, dump_ir=self.dump_ir)
        time = timeit.timeit(
            lambda: tuned_fun(*self.inputs), number=self.tuned_func_evals  # pylint: disable=cell-var-from-loop
        )
        if time < best_time:
          best_time = time
          best_fun = tuned_fun
          best_schedule = meta_schedule
      except:
        time = math.inf

      previous_configs.append(TuningResult(config, time))
      times.append(time)

    return best_time, best_fun, best_schedule, times

  def get_tuning_vars(self, module: ir.Module) -> list[ir.Operation]:
    """Returns a list of all tuning parameters in a module."""
    tuning_vars: list[ir.Operation] = []
    _walk(
        module,
        lambda op: tuning_vars.append(op)
        if op.operation.name == "transform.jasc.tuning_param"
        else None,
    )
    return tuning_vars


@dataclasses.dataclass
class FooTuner(TunerBase):
  """Example implementation of a custom tuner.

  New tuning configurations are determined by successively adding 1 to the
  default config of each tuning variable.
  """

  def get_tuning_config(
      self,
      tuning_vars: Sequence[jasc_transform_ops.TuningParamOp],
      previous_results: Sequence[TuningResult],
  ) -> TuningConfig:
    config: list[int] = []
    if len(previous_results) == 0:
      for op in tuning_vars:
        config.append(op.default_value.value)
    else:
      config = [
          config_val + 1
          for config_val in previous_results[-1].configuration.values
      ]
    return TuningConfig(config)


@dataclasses.dataclass(frozen=True)
class TuningResult:
  configuration: TuningConfig
  result: float


@dataclasses.dataclass
class TuningConfig:
  values: list[int]


def _walk(op: ir.Operation, callback: Callable[[ir.Operation], None]) -> None:
  """Calls the `callback` function on `op` and recurses to all nested ops."""
  callback(op)
  for region in op.operation.regions:
    for block in region.blocks:
      for nested_op in block.operations:
        _walk(nested_op, callback)
