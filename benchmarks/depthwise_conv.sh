#!/bin/bash

set -ex

# AVX512 throttling needs a lot of iteration, only report the last 100 after
# throttling has had a good chance of happening.
export SANDBOX_KEEP_LAST_N_RUNS=100

export BASE_SCRIPT_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source ${BASE_SCRIPT_PATH}/benchmark.sh

function depthwise_conv_1d_l1_repro() {
  COMMAND="cset proc -s sandbox_0 -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"

  STRIDES_AND_DILATIONS='[1],[1]'
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,8,32,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,16,32,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,20,32,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,32,32,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,40,32,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,55,32,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,64,32,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,80,32,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D6x16x3Peel --problem_sizes_list 1,96,32,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,110,32,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,128,32,3,[1],[1] --n_iters=3000)

  STRIDES_AND_DILATIONS='[2],[1]'
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,8,32,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D8x16x3Peel --problem_sizes_list 1,16,32,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,20,32,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,32,32,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,40,32,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,55,32,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D6x16x3Peel --problem_sizes_list 1,64,32,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,80,32,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D6x16x3Peel --problem_sizes_list 1,96,32,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D8x16x3Peel --problem_sizes_list 1,110,32,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D8x16x3Peel --problem_sizes_list 1,128,32,3,[2],[1] --n_iters=3000)

  STRIDES_AND_DILATIONS='[1],[2]'
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,8,32,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,16,32,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,20,32,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,32,32,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,40,32,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,55,32,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,64,32,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D6x16x3Peel --problem_sizes_list 1,80,32,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,96,32,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,110,32,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D8x16x3Peel --problem_sizes_list 1,128,32,3,[1],[2] --n_iters=3000)

  STRIDES_AND_DILATIONS='[2],[2]'
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,8,32,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,16,32,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,20,32,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,32,32,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,40,32,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,55,32,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,64,32,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D6x16x3Peel --problem_sizes_list 1,80,32,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D8x16x3Peel --problem_sizes_list 1,96,32,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D8x16x3Peel --problem_sizes_list 1,110,32,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D8x16x3Peel --problem_sizes_list 1,128,32,3,[2],[2] --n_iters=3000)
}


function run_depthwise_conv_1d_l1_benchmarks_n_times() {
  if (test -z "$1") || (test -z "$2")
  then
    echo Usage run_depthwise_conv_1d_l1_benchmarks_n_times target_benchmark_dir n_times
    exit 1
  fi

  for i in $(seq $2); do
    run_one_and_append_results_to_data depthwise_conv_1d_l1_repro $1
  done

  echo To create plots of the results run a command such as: 
  echo python ./tools/plot_benchmark.py \
    --input ${BENCH_DIR}/all.data \
    --output ${BENCH_DIR}/all.pdf \
    --plot_name \"...\" \
    --metric_to_plot \"gbyte_per_s_per_iter\" \
    --benchmarks_to_plot \"...\"
}




function depthwise_conv_1d_l2_repro() {
  COMMAND="cset proc -s sandbox_0 -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"

  STRIDES_AND_DILATIONS='[1],[1]'
  (${COMMAND} --expert_list SingleTiling3D6x16x3Peel --problem_sizes_list 1,16,256,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D8x16x3Peel --problem_sizes_list 1,20,256,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,32,256,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,40,256,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,55,256,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,64,256,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,80,256,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,96,256,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D6x16x3Peel --problem_sizes_list 1,110,256,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,128,256,3,[1],[1] --n_iters=3000)

  STRIDES_AND_DILATIONS='[2],[1]'
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,16,256,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,20,256,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D6x16x3Peel --problem_sizes_list 1,32,256,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D6x16x3Peel --problem_sizes_list 1,40,256,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D8x16x3Peel --problem_sizes_list 1,55,256,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,64,256,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,80,256,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,96,256,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,110,256,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,128,256,3,[2],[1] --n_iters=3000)

  STRIDES_AND_DILATIONS='[1],[2]'
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,16,256,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D6x16x3Peel --problem_sizes_list 1,20,256,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,32,256,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D8x16x3Peel --problem_sizes_list 1,40,256,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,55,256,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D6x16x3Peel --problem_sizes_list 1,64,256,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,80,256,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D6x16x3Peel --problem_sizes_list 1,96,256,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,110,256,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,128,256,3,[1],[2] --n_iters=3000)

  STRIDES_AND_DILATIONS='[2],[2]'
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,16,256,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,20,256,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,32,256,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D6x16x3Peel --problem_sizes_list 1,40,256,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,55,256,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,64,256,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,80,256,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,96,256,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,110,256,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,128,256,3,[2],[2] --n_iters=3000)
}

function run_depthwise_conv_1d_l2_benchmarks_n_times() {
  if (test -z "$1") || (test -z "$2")
  then
    echo Usage run_depthwise_conv_1d_l2_benchmarks_n_times target_benchmark_dir n_times
    exit 1
  fi

  for i in $(seq $2); do
    run_one_and_append_results_to_data depthwise_conv_1d_l2_repro $1
  done

  echo To create plots of the results run a command such as: 
  echo python ./tools/plot_benchmark.py \
    --input ${BENCH_DIR}/all.data \
    --output ${BENCH_DIR}/all.pdf \
    --plot_name \"...\" \
    --metric_to_plot \"gbyte_per_s_per_iter\" \
    --benchmarks_to_plot \"...\"
}




function depthwise_conv_1d_l3_repro() {
  COMMAND="cset proc -s sandbox_0 -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"

  STRIDES_AND_DILATIONS='[1],[1]'
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,256,1024,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,320,1024,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,512,1024,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,640,1024,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,880,1024,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,1024,1024,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,1280,1024,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,1536,1024,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,1760,1024,3,[1],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D8x16x3Peel --problem_sizes_list 1,2048,1024,3,[1],[1] --n_iters=3000)

  STRIDES_AND_DILATIONS='[2],[1]'
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,256,1024,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,320,1024,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,512,1024,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,640,1024,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,880,1024,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,1024,1024,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,1280,1024,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D6x16x3Peel --problem_sizes_list 1,1536,1024,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,1760,1024,3,[2],[1] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D8x16x3Peel --problem_sizes_list 1,2048,1024,3,[2],[1] --n_iters=3000)

  STRIDES_AND_DILATIONS='[1],[2]'
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,256,1024,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,320,1024,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,512,1024,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,640,1024,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,880,1024,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,1024,1024,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,1280,1024,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,1536,1024,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,1760,1024,3,[1],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D6x16x3Peel --problem_sizes_list 1,2048,1024,3,[1],[2] --n_iters=3000)

  STRIDES_AND_DILATIONS='[2],[2]'
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,256,1024,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,320,1024,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,512,1024,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,640,1024,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,880,1024,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,1024,1024,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,1280,1024,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,1536,1024,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D5x16x3Peel --problem_sizes_list 1,1760,1024,3,[2],[2] --n_iters=3000)
  (${COMMAND} --expert_list SingleTiling3D4x16x3Peel --problem_sizes_list 1,2048,1024,3,[2],[2] --n_iters=3000)
}



function run_depthwise_conv_1d_l3_benchmarks_n_times() {
  if (test -z "$1") || (test -z "$2")
  then
    echo Usage run_depthwise_conv_1d_l3_benchmarks_n_times target_benchmark_dir n_times
    exit 1
  fi

  for i in $(seq $2); do
    run_one_and_append_results_to_data depthwise_conv_1d_l3_repro $1
  done

  echo To create plots of the results run a command such as: 
  echo python ./tools/plot_benchmark.py \
    --input ${BENCH_DIR}/all.data \
    --output ${BENCH_DIR}/all.pdf \
    --plot_name \"...\" \
    --metric_to_plot \"gbyte_per_s_per_iter\" \
    --benchmarks_to_plot \"...\"
}


function depthwise_conv_2d_mobilenet() {
  COMMAND="cset proc -s sandbox_0 -e python -- -m python.examples.depthwise_conv.depthwise_conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"

  (${COMMAND} --expert_list DoubleTileAndDecompose8x14x32then7x32x1x3 --problem_sizes_list 1,112,112,32,3,3,[1,1],[1,1] --n_iters=3000)
  (${COMMAND} --expert_list DoubleTileAndDecompose4x14x32then8x32x1x3 --problem_sizes_list 1,56,56,128,3,3,[1,1],[1,1] --n_iters=3000)
  (${COMMAND} --expert_list DoubleTileAndDecompose8x14x32then7x32x1x3 --problem_sizes_list 1,56,56,128,3,3,[2,2],[1,1] --n_iters=3000)
  (${COMMAND} --expert_list DoubleTileAndDecompose8x14x32then7x32x1x3 --problem_sizes_list 1,28,28,256,3,3,[1,1],[1,1] --n_iters=3000)
  (${COMMAND} --expert_list DoubleTileAndDecompose8x14x32then7x32x1x3 --problem_sizes_list 1,28,28,256,3,3,[2,2],[1,1] --n_iters=3000)
  (${COMMAND} --expert_list DoubleTileAndDecompose8x14x32then7x32x1x3 --problem_sizes_list 1,14,14,512,3,3,[1,1],[1,1] --n_iters=3000)
  (${COMMAND} --expert_list DoubleTileAndDecompose4x14x32then8x32x1x3 --problem_sizes_list 1,14,14,512,3,3,[2,2],[1,1] --n_iters=3000)
  (${COMMAND} --expert_list DoubleTileAndDecompose8x14x32then7x32x1x3 --problem_sizes_list 1,7,7,1024,3,3,[1,1],[1,1] --n_iters=3000)
}

function run_depthwise_conv_2d_mobilenet_benchmarks_n_times() {
  if (test -z "$1") || (test -z "$2")
  then
    echo Usage run_depthwise_conv_2d_mobilenet_benchmarks_n_times target_benchmark_dir n_times
    exit 1
  fi

  for i in $(seq $2); do
    run_one_and_append_results_to_data depthwise_conv_2d_mobilenet $1
  done

  echo To create plots of the results run a command such as: 
  echo python ./tools/plot_benchmark.py \
    --input ${BENCH_DIR}/all.data \
    --output ${BENCH_DIR}/all.pdf \
    --plot_name \"...\" \
    --metric_to_plot \"gbyte_per_s_per_iter\" \
    --benchmarks_to_plot \"...\"
}