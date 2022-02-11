#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from examples.core.transforms import *
from examples.core.transform import TransformListFactory, TransformationList


class Pipeline(Transform):
  """Tile a linalg op with `tile_sizes`.

    This transform can be configured as follows:
    * `ms_unroll`: Level of unrolling of the given loop
    * `ms_distance`: Distance between a load and a compute operation
    """

  variables = {
      "unroll": (IntVariable, []),
      "distance": (IntVariable, []),
  }

  def __init__(self, fun_name: str, op_name: str, **kwargs):
    self._parse_variables_in_kwargs(kwargs)
    unrolling_str = f"unrolling={self.unroll}"
    distance_str = f"distance={self.distance}"
    pipeline = (f"alp-modulo-scheduling{{"
                f"     {unrolling_str} "
                f"     {distance_str}}},"
                f"canonicalize,"
                f"cse")
    self.pipeline = f"builtin.func({pipeline})"


class ExtractKernel(Transform):
  """Tile a linalg op with `tile_sizes`.

    This transform can be configured as follows:
    * `ms_unroll`: Level of unrolling of the given loop
    * `ms_distance`: Distance between a load and a compute operation
    """

  def __init__(self, fun_name: str, op_name: str, **kwargs):
    self.pipeline = f"alp-extract-kernel," f"canonicalize," f"cse"


class ConvertLoops(Transform):
  """Tile a linalg op with `tile_sizes`.

    This transform can be configured as follows:
    * `ms_unroll`: Level of unrolling of the given loop
    * `ms_distance`: Distance between a load and a compute operation
    """

  def __init__(self, fun_name: str, op_name: str, **kwargs):
    self._parse_variables_in_kwargs(kwargs)
    pipeline = (f"alp-for-to-dowhile{{"
                f"     anchor-func={fun_name}}},"
                f"canonicalize,"
                f"cse")
    self.pipeline = f"builtin.func({pipeline})"
