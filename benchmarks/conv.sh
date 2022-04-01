#!/bin/bash

set -ex

# AVX512 throttling needs a lot of iteration, only report the last 100 after
# throttling has had a good chance of happening.
export SANDBOX_KEEP_LAST_N_RUNS=100

export BASE_SCRIPT_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source ${BASE_SCRIPT_PATH}/benchmark.sh

function conv_1d_repro() {
  COMMAND="cset proc -s sandbox_parallel -e python -- -m python.examples.conv.conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"

  STRIDES_AND_DILATIONS='[1],[1]'
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,8,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,16,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,20,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list DoubleTile3DPeel --problem_sizes_list 1,32,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,40,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,55,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list DoubleTile3DPeel --problem_sizes_list 1,64,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)

  STRIDES_AND_DILATIONS='[2],[1]'
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,8,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,16,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,20,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,32,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list DoubleTile3DPad --problem_sizes_list 1,40,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,55,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list DoubleTile3DPeel --problem_sizes_list 1,64,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)

  STRIDES_AND_DILATIONS='[1],[2]'
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,8,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list DoubleTile3DPeel --problem_sizes_list 1,16,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,20,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list DoubleTile3DPeel --problem_sizes_list 1,32,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,40,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,55,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,64,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)

  STRIDES_AND_DILATIONS='[2],[2]'
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,8,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list DoubleTile3DPeel --problem_sizes_list 1,16,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,20,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list DoubleTile3DPeel --problem_sizes_list 1,32,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,40,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,55,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list DoubleTile3DPeel --problem_sizes_list 1,64,32,3,64,${STRIDES_AND_DILATIONS} --n_iters=1000)
}

function conv_2d_repro() {
  COMMAND="cset proc -s sandbox_parallel -e python -- -m python.examples.conv.conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"

  STRIDES_AND_DILATIONS='[1,1],[1,1]'
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,8,8,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,16,16,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,20,20,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,32,32,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list DoubleTile3DPeel --problem_sizes_list 1,40,40,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,55,55,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,64,64,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)

  STRIDES_AND_DILATIONS='[2,2],[1,1]'
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,8,8,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,16,16,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,20,20,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,32,32,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list DoubleTile3DPeel --problem_sizes_list 1,40,40,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list DoubleTile3DPeel --problem_sizes_list 1,55,55,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,64,64,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)

  STRIDES_AND_DILATIONS='[1,1],[2,2]'
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,8,8,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,16,16,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list DoubleTile3DPeel --problem_sizes_list 1,20,20,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list DoubleTile3DPeel --problem_sizes_list 1,32,32,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,40,40,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,55,55,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list DoubleTile3DPeel --problem_sizes_list 1,64,64,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)

  STRIDES_AND_DILATIONS='[2,2],[2,2]'
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,8,8,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,16,16,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,20,20,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list DoubleTile3DPeel --problem_sizes_list 1,32,32,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 1,40,40,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list DoubleTile3DPeel --problem_sizes_list 1,55,55,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
  (${COMMAND} --expert_list DoubleTile3DPeel --problem_sizes_list 1,64,64,32,3,3,64${STRIDES_AND_DILATIONS} --n_iters=1000)
}

function run_conv_1d_benchmarks_n_times() {
  if (test -z "$1") || (test -z "$2")
  then
    echo Usage run_conv_1d_benchmarks_n_times target_benchmark_dir n_times
    exit 1
  fi

  for i in $(seq $2); do
    run_one_and_append_results_to_data conv_1d_repro $1
  done

  echo To create plots of the results run a command such as: 
  echo python ./tools/plot_benchmark.py \
    --input ${BENCH_DIR}/all.data \
    --output ${BENCH_DIR}/all.pdf \
    --plot_name \"XXXX\" \
    --metric_to_plot \"gbyte_per_s_per_iter\" \
    --benchmarks_to_plot \"matmul_mkkn\"
}

function run_conv_2d_benchmarks_n_times() {
  if (test -z "$1") || (test -z "$2")
  then
    echo Usage run_conv_2d_benchmarks_n_times target_benchmark_dir n_times
    exit 1
  fi

  for i in $(seq $2); do
    run_one_and_append_results_to_data conv_2d_repro $1
  done

  echo To create plots of the results run a command such as: 
  echo python ./tools/plot_benchmark.py \
    --input ${BENCH_DIR}/all.data \
    --output ${BENCH_DIR}/all.pdf \
    --plot_name \"XXXX\" \
    --metric_to_plot \"gbyte_per_s_per_iter\" \
    --benchmarks_to_plot \"matmul_mkkn\"
}