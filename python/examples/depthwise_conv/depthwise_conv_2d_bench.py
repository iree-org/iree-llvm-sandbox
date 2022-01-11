# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from .definitions import *

fun_name = 'depthwise_conv_2d_nhwc_hwc'
op_name = 'linalg.depthwise_conv_2d_nhwc_hwc'

################################################################################
# Compilation strategies.
################################################################################

all_names = ["DepthWiseConv2DExpert"]
all_experts = [
    # Note: `\` char at the end of next line prevents formatter reflows, keep it.
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [        \
        Tile(fun_name=fun_name,
             op_name=op_name,
             #           N   H   W   C KH, KW
             tile_sizes=[1, 8, 14, 32],
             peel=[0, 1, 2])
        .then(Tile(fun_name=fun_name,
                   op_name=op_name,
                   #           N  H  W   C KH, KW
                   tile_sizes=[1, 1, 7, 32, 1, 3],
                   peel=[0, 1, 2]))
        .then(DecomposeToLowerDimensionalNamedOp(fun_name=fun_name,
                                                 op_name=op_name))
        .then(Vectorize(fun_name, "linalg.depthwise_conv_1d_nwc_wc"))
        .then(Bufferize())
        .then(LowerVectors())
        .then(LowerToLLVM())
    ]
]

################################################################################
# Problem instantiation
################################################################################

keys = ['N', 'H', 'W', 'C', 'KH', 'KW', 'strides', 'dilations']


# CHECK-NOT: FAILURE
def main():
  n_iters = 1000
  #   N   H   W   C  KH  KW      st      dil
  microbenchmark_problem_size_list = [
      [1, 16, 16, 32, 3, 3, [1, 1], [1, 1]],
      [1, 16, 16, 32, 3, 3, [1, 2], [1, 2]],
      [1, 16, 16, 32, 3, 3, [2, 1], [1, 2]],
      [1, 16, 16, 32, 3, 3, [2, 2], [2, 2]],
      [1, 16, 16, 32, 3, 3, [2, 3], [3, 2]],
      [1, 16, 16, 32, 3, 3, [3, 2], [2, 3]],
  ]

  benchmark_problem_size_list = [
      ####################################################
      #   /*         H    W   KH  KW  PH  PW  S  D    G */
      ####################################################
      #   b->Args({112, 112,  3,  3,  2,  2, 1, 1,   32});
      #   b->Args({ 56,  56,  3,  3,  2,  2, 1, 1,  128});
      #   b->Args({ 56,  56,  3,  3,  2,  2, 2, 1,  128});
      #   b->Args({ 28,  28,  3,  3,  2,  2, 1, 1,  256});
      #   b->Args({ 28,  28,  3,  3,  2,  2, 2, 1,  256});
      #   b->Args({ 14,  14,  3,  3,  2,  2, 1, 1,  512});
      #   b->Args({ 14,  14,  3,  3,  2,  2, 2, 1,  512});
      ####################################################
      # N   H    W     C  KH  KW      st     dil
      ####################################################
      [1, 112, 112, 32, 3, 3, [1, 1], [1, 1]],
      [1, 56, 56, 128, 3, 3, [1, 1], [1, 1]],
      [1, 56, 56, 128, 3, 3, [2, 2], [1, 1]],
      [1, 28, 28, 256, 3, 3, [1, 1], [1, 1]],
      [1, 28, 28, 256, 3, 3, [2, 2], [1, 1]],
      [1, 14, 14, 512, 3, 3, [1, 1], [1, 1]],
      [1, 14, 14, 512, 3, 3, [2, 2], [1, 1]],
      [1, 7, 7, 1024, 3, 3, [1, 1], [1, 1]],
  ]

  # Specify default configuration and parse command line.
  args = test_argparser("depthwise conv 2d benchmark",
                        default_problem_sizes_list=benchmark_problem_size_list,
                        default_expert_list=all_names,
                        default_dynamic_at_compile_time_list=[],
                        default_spec_list=[])

  def numpy_kernel(args, sizes, types):
    problem = DepthwiseConvolutionProblem('NHWC',
                                          'HWC',
                                          strides=sizes['strides'],
                                          dilations=sizes['dilations'])
    problem.reference_np(*args)

  def pytorch_kernel(args, sizes, types):
    problem = DepthwiseConvolutionProblem('NHWC',
                                          'HWC',
                                          strides=sizes['strides'],
                                          dilations=sizes['dilations'])
    problem.reference_pt(*args)

  test_harness(
      lambda sizes, t: DepthwiseConvolutionProblem(
          'NHWC', 'HWC', strides=sizes['strides'], dilations=sizes['dilations'
                                                                  ]),
      [[np.float32] * 3],
      test_sizes(keys, args.problem_sizes_list),
      test_experts(all_experts, all_names, args.expert_list),
      n_iters=n_iters,
      function_name=fun_name,
      dump_ir_to_file='/tmp/abcd.mlir',
      dump_obj_to_file='/tmp/abcd.o',
      numpy_benchmark=numpy_kernel,
      pytorch_benchmark=pytorch_kernel,
      plot_path=args.plot_path)


if __name__ == '__main__':
  main()
