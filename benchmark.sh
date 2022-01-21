#!/bin/bash

export IREE_LLVM_SANDBOX_BUILD_DIR=$(dirname $0)/build 
export MLIR_RUNNER_UTILS_LIB=${IREE_LLVM_SANDBOX_BUILD_DIR}/lib/libmlir_runner_utils.so 
export MLIR_C_RUNNER_UTILS_LIB=${IREE_LLVM_SANDBOX_BUILD_DIR}/lib/libmlir_c_runner_utils.so 
export PYTHONPATH=${PYTHONPATH}:${IREE_LLVM_SANDBOX_BUILD_DIR}/tools/sandbox/python_package 
export PATH=$PATH:$(dirname ~ntv)/.venv/mlirdev/bin/

# On my machine (theoretical peak 384 GB/s L1 BW) I see:
# [100,  32],    # sweet spot for prefetchers, seems to maximize L1 BW @ 281GB/s
#
# [ 50, 272],    # 10% L2 load, L2 BW @ 71.6GB/s
# [100, 272],    # 20% L2 load, L2 BW @ 80.8GB/s
# [150, 272],    # 30% L2 load, L2 BW @ 69.3GB/s
# [200, 272],    # 40% L2 load, L2 BW @ 82GB/s
# [250, 272],    # 50% L2 load, L2 BW @ 81GB/s
# [300, 272],    # 60% L2 load, L2 BW @ 76GB/s
# [350, 272],    # 70% L2 load, L2 BW @ 65.3GB/s
# [400, 272],    # 80% L2 load, L2 BW @ 56.5GB/s
# [450, 272],    # 90% L2 load, L2 BW @ 54.8GB/s
# [500, 272],    # 100% L2 load, L2 BW @ 47.7GB/s
#
# [5000, 272],   # 40% L3 load, L3 BW @ 25.7GB/s
# [10000, 272],  # 80% L3 load, L3 BW @ 17.2GB/s
# [15000, 272],  # 120% L3 load, L3 BW @ 10.8GB/s
#
# [300000, 272], # DRAM (24x L3 load), L3 BW @ 12GB/s
function copy_benchmark() {
  cset proc -s sandbox -e python -- -m python.examples.copy.copy_2d_bench
}

###############################################################################
# Some static matmul mk,kn benchmarks.
###############################################################################

function check_usage() {
  echo $1
  echo $2
  if (test -z "$1" ) || (test -z "$2" )
  then
    echo "Usage benchmark.sh data_filename plot_filename.pdf"
    exit
  fi
}

function matmul_static_small() {
  check_usage $1 $2
  # 179 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.matmul.bench --expert_list SingleTiling2DPeel --dynamic_at_compile_time_list [] --spec_list mk,kn --problem_sizes_list 18,32,96 --dump_data $1
  # 170 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.matmul.bench --expert_list SingleTiling2DPeel --dynamic_at_compile_time_list [] --spec_list mk,kn --problem_sizes_list 24,64,96 --dump_data $1
  # 172 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.matmul.bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --spec_list mk,kn --problem_sizes_list 48,64,128 --dump_data $1
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "Matrix Multiplication Performance (Small Static Sizes)"
}

function matmul_static_small_reduction_dimension() {
  check_usage $1 $2
  # 93 GFlop/s -> NYI perf bug
  cset proc -s sandbox -e python -- -m python.examples.matmul.bench --expert_list SingleTiling2DPeel --dynamic_at_compile_time_list [] --spec_list mk,kn --problem_sizes_list 480,512,16 --dump_data $1
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "Matrix Multiplication Performance (Small Reduction Static Sizes)"
}

function matmul_static_medium() {
  check_usage $1 $2
  # 151 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.matmul.bench --expert_list SingleTiling3DPad --dynamic_at_compile_time_list [] --spec_list mk,kn --problem_sizes_list  384,256,256 --dump_data $1
  # 145 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.matmul.bench --expert_list SingleTiling3DPad --dynamic_at_compile_time_list [] --spec_list mk,kn --problem_sizes_list  480,512,256 --dump_data $1
  # 157 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.matmul.bench --expert_list SingleTiling2DPeel --dynamic_at_compile_time_list [] --spec_list mk,kn --problem_sizes_list  784,128,512 --dump_data $1
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "Matrix Multiplication Performance (Medium Static Sizes)"
}

function matmul_static_large() {
  check_usage $1 $2
  # 158 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.matmul.bench --expert_list DoubleTile2DPadAndHoist --dynamic_at_compile_time_list [] --spec_list mk,kn --problem_sizes_list  1020,1152,1152 --dump_data $1
  # 148 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.matmul.bench --expert_list DoubleTile2DPadAndHoist --dynamic_at_compile_time_list [] --spec_list mk,kn --problem_sizes_list  1920,2304,2304 --dump_data $1
  # 151 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.matmul.bench --expert_list DoubleTile2DPadAndHoist --dynamic_at_compile_time_list [] --spec_list mk,kn --problem_sizes_list  2304,2304,2560 --dump_data $1
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "Matrix Multiplication Performance (Large Static Sizes)"
}

###############################################################################
# Some static conv1d nwc benchmarks.
###############################################################################
# Batch size 1, 32 -> 64 channels, stride 1, dilation 1.
function conv_1d_static_small_stride_1_dilation_1() {
  check_usage $1 $2
  # 166 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,32,3,64,[1],[1] --dump_data $1
  # 167 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,32,3,64,[1],[1] --dump_data $1 
  # 150 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,32,3,64,[1],[1] --dump_data $1 
  # 142 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,32,3,64,[1],[1] --dump_data $1 
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "1D Convolution Performance (Small Static Sizes)"
}

# Batch size 1, 128 -> 256 channels, stride 1, dilation 1.
function conv_1d_static_medium_stride_1_dilation_1() {
  check_usage $1 $2
  # 166 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,128,3,256,[1],[1] --dump_data $1
  # 152 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,128,3,256,[1],[1] --dump_data $1 
  # 141 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,128,3,256,[1],[1] --dump_data $1 
  # 150 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,128,3,256,[1],[1] --dump_data $1 
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "1D Convolution Performance (Medium Static Sizes)"
}

# Batch size 1, 512 -> 1024 channels, stride 1, dilation 1.
function conv_1d_static_large_stride_1_dilation_1() {
  check_usage $1 $2
  # 101 GFlop/s -> NYI perf bug
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list DoubleTile3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,512,3,1024,[1],[1] --dump_data $1
  # 98 GFlop/s -> NYI perf bug
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench  --expert_list DoubleTile3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,512,3,1024,[1],[1] --dump_data $1 
  # 97 GFlop/s -> NYI perf bug
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list DoubleTile3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,512,3,1024,[1],[1] --dump_data $1 
  # 95 GFlop/s -> NYI perf bug
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list DoubleTile3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,512,3,1024,[1],[1] --dump_data $1 
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "1D Convolution Performance (Large Static Sizes)"
}

# Batch size 1, 32 -> 64 channels, stride 2, dilation 1.
function conv_1d_static_small_stride_2_dilation_1() {
  check_usage $1 $2
  # 136 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,32,3,64,[2],[1] --dump_data $1
  # 136 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,32,3,64,[2],[1] --dump_data $1 
  # 120 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,32,3,64,[2],[1] --dump_data $1 
  # 118 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,32,3,64,[2],[1] --dump_data $1 
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "1D Convolution Performance (Small Static Sizes)"
}

# Batch size 1, 128 -> 256 channels, stride 2, dilation 1.
function conv_1d_static_medium_stride_2_dilation_1() {
  check_usage $1 $2
  # 125 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,128,3,256,[2],[1] --dump_data $1
  # 118 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,128,3,256,[2],[1] --dump_data $1 
  # 125 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,128,3,256,[2],[1] --dump_data $1 
  # 123 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,128,3,256,[2],[1] --dump_data $1 
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "1D Convolution Performance (Small Static Sizes)"
}

# Batch size 1, 512 -> 1024 channels, stride 2, dilation 1.
function conv_1d_static_large_stride_2_dilation_1() {
  check_usage $1 $2
  # 80 GFlop/s -> NYI perf bug
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list DoubleTile3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,512,3,1024,[2],[1] --dump_data $1
  # 81 GFlop/s -> NYI perf bug
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench  --expert_list  DoubleTile3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,512,3,1024,[2],[1] --dump_data $1 
  # 80 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list  DoubleTile3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,512,3,1024,[2],[1] --dump_data $1 
  # 80 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list  DoubleTile3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,512,3,1024,[2],[1] --dump_data $1 
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "1D Convolution Performance (Small Static Sizes)"
}

# Batch size 1, 32 -> 64 channels, stride 1, dilation 2.
function conv_1d_static_small_stride_1_dilation_2() {
  check_usage $1 $2
  # 168 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,32,3,64,[1],[2] --dump_data $1
  # 167 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,32,3,64,[1],[2] --dump_data $1 
  # 149 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,32,3,64,[1],[2] --dump_data $1 
  # 145 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,32,3,64,[1],[2] --dump_data $1 
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "1D Convolution Performance (Small Static Sizes)"
}

# Batch size 1, 128 -> 256 channels, stride 1, dilation 2.
function conv_1d_static_medium_stride_1_dilation_2() {
  check_usage $1 $2
  # 156 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,128,3,256,[1],[2] --dump_data $1
  # 154 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,128,3,256,[1],[2] --dump_data $1 
  # 151 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,128,3,256,[1],[2] --dump_data $1 
  # 150 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,128,3,256,[1],[2] --dump_data $1 
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "1D Convolution Performance (Small Static Sizes)"
}

# Batch size 1, 512 -> 1024 channels, stride 1, dilation 2.
function conv_1d_static_large_stride_1_dilation_2() {
  check_usage $1 $2
  # 103 GFlop/s -> NYI perf bug
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list DoubleTile3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,512,3,1024,[1],[2] --dump_data $1
  # 99 GFlop/s -> NYI perf bug
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench  --expert_list  DoubleTile3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,512,3,1024,[1],[2] --dump_data $1 
  # 101 GFlop/s -> NYI perf bug
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list  DoubleTile3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,512,3,1024,[1],[2] --dump_data $1 
  # 97 GFlop/s -> NYI perf bug
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list  DoubleTile3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,512,3,1024,[1],[2] --dump_data $1 
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "1D Convolution Performance (Small Static Sizes)"
}

###############################################################################
# Some static conv2d nhwc benchmarks.
###############################################################################

# Batch size 1, 32 -> 64 channels, stride [1, 1], dilation [1, 1].
function conv_2d_static_small_stride_1_dilation_1() {
  check_usage $1 $2
  # 180 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_2d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,16,16,32,3,3,64,[1,1],[1,1] --dump_data $1
  # 173 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_2d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,26,38,32,3,3,64,[1,1],[1,1] --dump_data $1
  # 163 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_2d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,56,74,32,3,3,64,[1,1],[1,1] --dump_data $1
  # 165 GFlop/s
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_2d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,100,113,32,3,3,64,[1,1],[1,1] --dump_data $1
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "2D Convolution Performance (Small Static Sizes)"
}

# Batch size 1, 128 -> 256 channels, stride [1, 1], dilation [1, 1].
function conv_2d_static_medium_stride_1_dilation_1() {
  check_usage $1 $2
  # 75 GFlop/s -> NYI perf bug
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_2d_bench --expert_list DoubleTile3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,16,16,128,3,3,256,[1,1],[1,1] --dump_data $1
  # 74 GFlop/s -> NYI perf bug
  cset proc -s sandbox -e python -- -m python.examples.conv.conv_2d_bench --expert_list DoubleTile3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,26,38,128,3,3,256,[1,1],[1,1] --dump_data $1
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "2D Convolution Performance (Medium Static Sizes)"
}



###############################################################################
# Some static depthwise_conv_1d nhwc benchmarks.
###############################################################################
# Batch size 1, 32 channels, stride 1, dilation 1.
function depthwise_conv_1d_static_small_stride_1_dilation_1() {
  check_usage $1 $2
  # 53 GFlop/s 107 GB/s
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,32,3,[1],[1] --dump_data $1
  # 59 GFlop/s 118 GB/s
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,32,3,[1],[1] --dump_data $1
  # 37 GFlop/s 75 GB/s
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,32,3,[1],[1] --dump_data $1
  # 19 GFlop/s 38 GB/s
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,32,3,[1],[1] --dump_data $1
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "1D Depthwise Convolution Performance (Small Static Sizes)"
}
# Batch size 1, 32 channels, stride 2, dilation 1.
function depthwise_conv_1d_static_small_stride_2_dilation_1() {
  check_usage $1 $2
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,32,3,[2],[1] --dump_data $1
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,32,3,[2],[1] --dump_data $1
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,32,3,[2],[1] --dump_data $1
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,32,3,[2],[1] --dump_data $1
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "1D Depthwise Convolution Performance (Small Static Sizes)"
}
# Batch size 1, 32 channels, stride 1, dilation 2.
function depthwise_conv_1d_static_small_stride_1_dilation_2() {
  check_usage $1 $2
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,32,3,[1],[2] --dump_data $1
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,32,3,[1],[2] --dump_data $1
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,32,3,[1],[2] --dump_data $1
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,32,3,[1],[2] --dump_data $1
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "1D Depthwise Convolution Performance (Small Static Sizes)"
}

# Batch size 1, 128 channels, stride 1, dilation 1.
function depthwise_conv_1d_static_medium_stride_1_dilation_1() {
  check_usage $1 $2
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,128,3,[1],[1] --dump_data $1
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,128,3,[1],[1] --dump_data $1
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,128,3,[1],[1] --dump_data $1
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,128,3,[1],[1] --dump_data $1
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "1D Depthwise Convolution Performance (Medium Static Sizes)"
}
# Batch size 1, 128 channels, stride 2, dilation 1.
function depthwise_conv_1d_static_medium_stride_2_dilation_1() {
  check_usage $1 $2
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,128,3,[2],[1] --dump_data $1
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,128,3,[2],[1] --dump_data $1
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,128,3,[2],[1] --dump_data $1
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,128,3,[2],[1] --dump_data $1
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "1D Depthwise Convolution Performance (Medium Static Sizes)"
}
# Batch size 1, 128 channels, stride 1, dilation 2.
function depthwise_conv_1d_static_medium_stride_1_dilation_2() {
  check_usage $1 $2
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,128,3,[1],[2] --dump_data $1
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,128,3,[1],[2] --dump_data $1
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,128,3,[1],[2] --dump_data $1
  cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,128,3,[1],[2] --dump_data $1
  python ./tools/plot_benchmark.py -i $1 -o $2 -n "1D Depthwise Convolution Performance (Medium Static Sizes)"
}

  # "depthwise_conv_2d" : {
  #   "module" : "python.examples.depthwise_conv.depthwise_conv_2d_bench",
  #   "arguments" : {
  #     "n_iters" : 100,
  #     "problem_sizes_list" : [
  #       "8,16,16,32,3,3,[1,1],[1,1]", "8,16,16,32,3,3,[2,2],[2,2]",
  #       "8,26,38,32,3,3,[1,1],[1,1]", "8,26,38,32,3,3,[2,2],[2,2]",
  #       "8,56,74,32,3,3,[1,1],[1,1]", "8,56,74,32,3,3,[2,2],[2,2]",
  #       "8,100,113,32,3,3,[1,1],[1,1]", "8,100,113,32,3,3,[2,2],[2,2]"
  #     ],
  #   }
  # },
  # "row_reduction_2d" : {
  #   "module" : "python.examples.reduction.row_reduction_2d_bench",
  #   "arguments" : {
  #     "n_iters" : 100,
  #     "problem_sizes_list" : [
  #       "100,256",
  #       "200,512",
  #       "500,512",
  #       "500,1024",
  #       "1000,1024",
  #       "4000,6144",
  #       "8000,6144"
  #     ],
  #   }
  # },
  # "column_reduction_2d" : {
  #   "module" : "python.examples.reduction.column_reduction_2d_bench",
  #   "arguments" : {
  #     "n_iters" : 100,
  #     "problem_sizes_list" : [
  #       "100,256",
  #       "200,512",
  #       "500,512",
  #       "500,1024",
  #       "1000,1024",
  #       "4000,6144",
  #       "8000,6144"
  #     ],

function run_all() {
  BENCH_ROOT_DIR=$(dirname $0)/benchmarks
  BENCH_DIR=${BENCH_ROOT_DIR}/results_$(ls -l ${BENCH_ROOT_DIR} | wc -l)
  mkdir -p ${BENCH_DIR}

  matmul_static_small ${BENCH_DIR}/matmul_static_small.data ${BENCH_DIR}/matmul_static_small.pdf
  matmul_static_small_reduction_dimension ${BENCH_DIR}/matmul_static_small_reduction_dimension.data ${BENCH_DIR}/matmul_static_small_reduction_dimension.pdf
  matmul_static_medium ${BENCH_DIR}/matmul_static_medium.data ${BENCH_DIR}/matmul_static_medium.pdf
  matmul_static_large ${BENCH_DIR}/matmul_static_large.data ${BENCH_DIR}/matmul_static_large.pdf
  
  conv_1d_static_small_stride_1_dilation_1 ${BENCH_DIR}/conv_1d_static_small_stride_1_dilation_1.data ${BENCH_DIR}/conv_1d_static_small_stride_1_dilation_1.pdf
  conv_1d_static_medium_stride_1_dilation_1 ${BENCH_DIR}/conv_1d_static_medium_stride_1_dilation_1.data ${BENCH_DIR}/conv_1d_static_medium_stride_1_dilation_1.pdf
  conv_1d_static_large_stride_1_dilation_1 ${BENCH_DIR}/conv_1d_static_large_stride_1_dilation_1.data ${BENCH_DIR}/conv_1d_static_large_stride_1_dilation_1.pdf

  conv_1d_static_small_stride_2_dilation_1 ${BENCH_DIR}/conv_1d_static_small_stride_2_dilation_1.data ${BENCH_DIR}/conv_1d_static_small_stride_2_dilation_1.pdf
  conv_1d_static_medium_stride_2_dilation_1 ${BENCH_DIR}/conv_1d_static_medium_stride_2_dilation_1.data ${BENCH_DIR}/conv_1d_static_medium_stride_2_dilation_1.pdf
  conv_1d_static_large_stride_2_dilation_1 ${BENCH_DIR}/conv_1d_static_large_stride_2_dilation_1.data ${BENCH_DIR}/conv_1d_static_large_stride_2_dilation_1.pdf

  conv_1d_static_small_stride_1_dilation_2 ${BENCH_DIR}/conv_1d_static_small_stride_1_dilation_2.data ${BENCH_DIR}/conv_1d_static_small_stride_1_dilation_2.pdf
  conv_1d_static_medium_stride_1_dilation_2 ${BENCH_DIR}/conv_1d_static_medium_stride_1_dilation_2.data ${BENCH_DIR}/conv_1d_static_medium_stride_1_dilation_2.pdf
  conv_1d_static_large_stride_1_dilation_2 ${BENCH_DIR}/conv_1d_static_large_stride_1_dilation_2.data ${BENCH_DIR}/conv_1d_static_large_stride_1_dilation_2.pdf

  conv_2d_static_small_stride_1_dilation_1 ${BENCH_DIR}/conv_2d_static_small_stride_1_dilation_1.data ${BENCH_DIR}/conv_2d_static_small_stride_1_dilation_1.pdf
  conv_2d_static_medium_stride_1_dilation_1 ${BENCH_DIR}/conv_2d_static_medium_stride_1_dilation_1.data ${BENCH_DIR}/conv_2d_static_medium_stride_1_dilation_1.pdf

  depthwise_conv_1d_static_small_stride_1_dilation_1 ${BENCH_DIR}/depthwise_conv_1d_static_small_stride_1_dilation_1.data ${BENCH_DIR}/depthwise_conv_1d_static_small_stride_1_dilation_1.pdf
  depthwise_conv_1d_static_small_stride_2_dilation_1 ${BENCH_DIR}/depthwise_conv_1d_static_small_stride_2_dilation_1.data ${BENCH_DIR}/depthwise_conv_1d_static_small_stride_2_dilation_1.pdf
  depthwise_conv_1d_static_small_stride_1_dilation_2 ${BENCH_DIR}/depthwise_conv_1d_static_small_stride_1_dilation_2.data ${BENCH_DIR}/depthwise_conv_1d_static_small_stride_1_dilation_2.pdf

  depthwise_conv_1d_static_medium_stride_1_dilation_1 ${BENCH_DIR}/depthwise_conv_1d_static_medium_stride_1_dilation_1.data ${BENCH_DIR}/depthwise_conv_1d_static_medium_stride_1_dilation_1.pdf
  depthwise_conv_1d_static_medium_stride_2_dilation_1 ${BENCH_DIR}/depthwise_conv_1d_static_medium_stride_2_dilation_1.data ${BENCH_DIR}/depthwise_conv_1d_static_medium_stride_2_dilation_1.pdf
  depthwise_conv_1d_static_medium_stride_1_dilation_2 ${BENCH_DIR}/depthwise_conv_1d_static_medium_stride_1_dilation_2.data ${BENCH_DIR}/depthwise_conv_1d_static_medium_stride_1_dilation_2.pdf
}