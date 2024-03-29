# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Trampoline to run generated MLIR Python code.

Generated tablegen dialects expect to be able to find some symbols from the
mlir.dialects package.
"""
from jaxlib.mlir.dialects._ods_common import _cext, equally_sized_accessor, get_default_loc_context, get_op_result_or_op_results, get_op_result_or_value, get_op_results_or_values, segmented_accessor
