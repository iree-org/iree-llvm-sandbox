#!/bin/bash

export IREE_LLVM_SANDBOX_BUILD_DIR=$(dirname $0)/build 
export MLIR_RUNNER_UTILS_LIB=${IREE_LLVM_SANDBOX_BUILD_DIR}/lib/libmlir_runner_utils.so 
export MLIR_C_RUNNER_UTILS_LIB=${IREE_LLVM_SANDBOX_BUILD_DIR}/lib/libmlir_c_runner_utils.so 
export PYTHONPATH=${PYTHONPATH}:${IREE_LLVM_SANDBOX_BUILD_DIR}/tools/sandbox/python_package 
export PATH=$PATH:$(dirname ~ntv)/.venv/mlirdev/bin/

# Uncomment to benchmark bandwidth.
# cset proc -s sandbox -e python -- -m python.examples.copy.copy_2d_bench
# On my machine (theoretical peak 384 GB/s L1 BW) I see:
# [100, 32],  # sweet spot for prefetchers, seems to maximize L1 BW @ 295GB/s
# [100, 272],  # 10% L2 load, L2 BW @ 87GB/s
# [200, 272],  # 20% L2 load, L2 BW @ 84GB/s
# [300, 272],  # 30% L2 load, L2 BW @ 79GB/s
# [400, 272],  # 40% L2 load, L2 BW @ 73GB/s
# [500, 272],  # 50% L2 load, L2 BW @ 52GB/s
# [600, 272],  # 60% L2 load, L2 BW @ 35GB/s
# [700, 272],  # 70% L2 load, L2 BW @ 30GB/s
# [800, 272],  # 80% L2 load, L2 BW @ 30GB/s
# [900, 272],  # 90% L2 load, L2 BW @ 26.6GB/s
# [1000, 272],  # 100% L2 load, L2 BW @ 26.4GB/s
# [10000, 272],  # 40% L3 load, L3 BW @ 23.4GB/s
# [20000, 272],  # 80% L3 load, L3 BW @ 14.4GB/s
# [30000, 272],  # 120% L3 load, L3 BW @ 13GB/s
# [300000, 272],  # 12x L3 load, L3 BW @ 12GB/s

###############################################################################
# Some static matmul mk,kn benchmarks.
###############################################################################
# 179 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.matmul.bench --expert_list SingleTiling2DPeel --dynamic_at_compile_time_list [] --spec_list mk,kn --problem_sizes_list 18,32,96
# 170 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.matmul.bench --expert_list SingleTiling2DPeel --dynamic_at_compile_time_list [] --spec_list mk,kn --problem_sizes_list 24,64,96
# 172 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.matmul.bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --spec_list mk,kn --problem_sizes_list 48,64,128
# 93 GFlop/s -> NYI perf bug
# cset proc -s sandbox -e python -- -m python.examples.matmul.bench --expert_list SingleTiling2DPeel --dynamic_at_compile_time_list [] --spec_list mk,kn --problem_sizes_list 480,512,16
# 151 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.matmul.bench --expert_list SingleTiling3DPad --dynamic_at_compile_time_list [] --spec_list mk,kn --problem_sizes_list  384,256,256
# 145 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.matmul.bench --expert_list SingleTiling3DPad --dynamic_at_compile_time_list [] --spec_list mk,kn --problem_sizes_list  480,512,256
# 157 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.matmul.bench --expert_list SingleTiling2DPeel --dynamic_at_compile_time_list [] --spec_list mk,kn --problem_sizes_list  784,128,512
# 158 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.matmul.bench --expert_list DoubleTile2DPadAndHoist --dynamic_at_compile_time_list [] --spec_list mk,kn --problem_sizes_list  1020,1152,1152
# 148 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.matmul.bench --expert_list DoubleTile2DPadAndHoist --dynamic_at_compile_time_list [] --spec_list mk,kn --problem_sizes_list  1920,2304,2304
# 151 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.matmul.bench --expert_list DoubleTile2DPadAndHoist --dynamic_at_compile_time_list [] --spec_list mk,kn --problem_sizes_list  2304,2304,2560

###############################################################################
# Some static conv1d nwc benchmarks.
###############################################################################
# Batch size 1, 32 -> 64 channels, stride 1, dilation 1.
# 166 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,32,3,64,[1],[1]
# 167 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,32,3,64,[1],[1] 
# 150 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,32,3,64,[1],[1] 
# 142 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,32,3,64,[1],[1] 

# Batch size 1, 128 -> 256 channels, stride 1, dilation 1.
# 166 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,128,3,256,[1],[1]
# 152 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,128,3,256,[1],[1] 
# 141 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,128,3,256,[1],[1] 
# 150 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,128,3,256,[1],[1] 

# Batch size 1, 512 -> 1024 channels, stride 1, dilation 1.
# 101 GFlop/s -> NYI perf bug
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list DoubleTile3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,512,3,1024,[1],[1]
# 98 GFlop/s -> NYI perf bug
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench  --expert_list DoubleTile3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,512,3,1024,[1],[1] 
# 97 GFlop/s -> NYI perf bug
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list DoubleTile3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,512,3,1024,[1],[1] 
# 95 GFlop/s -> NYI perf bug
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list DoubleTile3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,512,3,1024,[1],[1] 


# Batch size 1, 32 -> 64 channels, stride 2, dilation 1.
# 136 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,32,3,64,[2],[1]
# 136 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,32,3,64,[2],[1] 
# 120 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,32,3,64,[2],[1] 
# 118 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,32,3,64,[2],[1] 

# Batch size 1, 128 -> 256 channels, stride 2, dilation 1.
# 125 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,128,3,256,[2],[1]
# 118 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,128,3,256,[2],[1] 
# 125 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,128,3,256,[2],[1] 
# 123 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list SingleTiling3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,128,3,256,[2],[1] 

# Batch size 1, 512 -> 1024 channels, stride 2, dilation 1.
# 80 GFlop/s -> NYI perf bug
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list DoubleTile3DPeel --dynamic_at_compile_time_list [] --problem_sizes_list 1,256,512,3,1024,[2],[1]
# 81 GFlop/s -> NYI perf bug
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench  --expert_list  DoubleTile3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,988,512,3,1024,[2],[1] 
# 80 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list  DoubleTile3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,4144,512,3,1024,[2],[1] 
# 80 GFlop/s
# cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench --expert_list  DoubleTile3DPad --dynamic_at_compile_time_list [] --problem_sizes_list 1,11300,512,3,1024,[2],[1] 
