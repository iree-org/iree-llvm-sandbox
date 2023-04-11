#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys
from pathlib import Path

import opentuner
from opentuner import ConfigurationManipulator
from opentuner.search.manipulator import (
    IntegerParameter,
    PowerOfTwoParameter,
    EnumParameter,
    BooleanParameter,
)
from opentuner import MeasurementInterface
from opentuner import Result

from . import mlirc
from .utils import parse
from ..benchmark import infra

max_flops = 0


class MLIRFlagsTuner(MeasurementInterface):

  def manipulator(self):
    """
        Define the search space by creating a
        ConfigurationManipulator
        """
    manipulator = ConfigurationManipulator()

    manipulator.add_parameter(PowerOfTwoParameter("mr", 4, 4))

    manipulator.add_parameter(PowerOfTwoParameter("nr", 16, 16))

    manipulator.add_parameter(PowerOfTwoParameter("kr", 16, 64))

    manipulator.add_parameter(PowerOfTwoParameter("kc", 64, 128))

    manipulator.add_parameter(PowerOfTwoParameter("mc", 256, 2048))

    manipulator.add_parameter(PowerOfTwoParameter("nc", 64, 2048))

    manipulator.add_parameter(IntegerParameter("ha", 4, 4))

    manipulator.add_parameter(IntegerParameter("hb", 3, 3))

    return manipulator

  def run(self, desired_result, input, limit):
    """
        Compile and run a given configuration then
        return performance
        """
    global max_flops
    cfg = desired_result.configuration.data

    mr = cfg["mr"]
    nr = cfg["nr"]
    kr = cfg["kr"]
    kc = cfg["kc"]
    mc = cfg["mc"]
    nc = cfg["nc"]
    ha = cfg["ha"]
    hb = cfg["hb"]

    reordering = "Afirst"

    if reordering == "Afirst":
      reorder_inner = [0, 1, 2]
      reorder_outer = [0, 2, 1]
    else:
      reorder_inner = [1, 0, 2]
      reorder_outer = [1, 2, 0]

    options = {
        f"tile_sizes": [mc, nc, kc],
        "register_tile_sizes": [mr, nr, 1],
        "split_vector_transfers_to": "vector-transfers",
        "unroll_vector_transfers": True,
        "reorder_tile_sizes": reorder_outer,
        "reorder_register_tile_sizes": reorder_inner,
        "hoist_packing": [ha, hb, 0],
        "transpose_packing": [0, 0, 0],
        "extract_micro_kernel": True,
        "modulo_scheduling": True,
        "ms_unroll": 1,
        "ms_distance": 1,
        "scheduler": "ilpmax",
        "verbosity_level": 0,
    }

    # Try to compile the program
    try:
      asm_program = mlirc.compile(self.args.input_file, options)
    except:
      return Result(time=sys.maxsize)

    # TODO: can we store obj_benchmark as an attribute of the class?
    mlir_benchmark = args.benchmark
    bench_base = os.path.splitext(mlir_benchmark)[0]
    obj_benchmark = bench_base + ".o"

    # Link and run
    exe = infra.link(asm_program, obj_benchmark)
    run_result = self.call_program(f"./{exe}")

    if run_result["returncode"] != 0:
      return Result(time=sys.maxsize)

    assert run_result["returncode"] == 0

    secs, flops = parse(run_result["stdout"])

    if flops > max_flops:
      max_flops = flops

    return Result(time=1 / flops)

  def save_final_config(self, configuration):
    """called at the end of tuning"""
    print("Optimal block size written to mmm_final_config.json:",
          configuration.data)
    self.manipulator().save_to_file(configuration.data, f"final_config.json")


# TODO: create an API to call tune() from the library packager
if __name__ == "__main__":
  argparser = opentuner.default_argparser()
  argparser.add_argument("--input-file", required=True)

  # TODO: is it possible to understand properties of the MLIR input file and
  # generating directly the benchmark program? Also, we should infer from the program
  # structure what transformations to run instead of having them hardcoded
  argparser.add_argument("--benchmark", required=True)

  args = argparser.parse_args()

  # Build the MLIR benchmark
  benchmark_obj = infra.compile(args.benchmark)

  MLIRFlagsTuner.main(args)
