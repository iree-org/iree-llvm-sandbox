#!/bin/bash

set -ex

# AVX512 throttling needs a lot of iteration, only report the last 100 after
# throttling has had a good chance of happening.
export SANDBOX_KEEP_LAST_N_RUNS=100

export BASE_SCRIPT_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source ${BASE_SCRIPT_PATH}/benchmark.sh

function matmul_mkkn_repro() {
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.matmul.bench --spec_list mk,kn ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list [] "

  (${COMMAND} --expert_list SingleTiling2DPeel --problem_sizes_list 18,32,96 --n_iters=200)
  (${COMMAND} --expert_list SingleTiling2DPeel --problem_sizes_list 24,64,96 --n_iters=200)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 48,64,128 --n_iters=200)
  (${COMMAND} --expert_list SingleTiling2DPeel --problem_sizes_list 192,64,128 --n_iters=200)
  (${COMMAND} --expert_list SingleTiling2DPeel --problem_sizes_list 192,128,128 --n_iters=200)
  (${COMMAND} --expert_list SingleTiling2DPeel --problem_sizes_list 480,512,16 --n_iters=200)
  (${COMMAND} --expert_list SingleTiling2DPeel --problem_sizes_list 384,256,256 --n_iters=200)
  (${COMMAND} --expert_list SingleTiling3DPad --problem_sizes_list 480,512,256 --n_iters=200)
  (${COMMAND} --expert_list SingleTiling2DPeel --problem_sizes_list 784,128,512 --n_iters=200)
  (${COMMAND} --expert_list DoubleTile2DPadAndHoist --problem_sizes_list 1020,1152,1152 --n_iters=200)
  (${COMMAND} --expert_list DoubleTile2DPadAndHoist --problem_sizes_list 1920,2304,2304 --n_iters=200)
}

function matmul_kmkn_repro() {
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.matmul.bench --spec_list km,kn ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list [] "

  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 18,32,96 --n_iters=200)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 24,64,96 --n_iters=200)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 48,64,128 --n_iters=200)
  (${COMMAND} --expert_list SingleTiling3DPeel --problem_sizes_list 192,64,128 --n_iters=200)
  (${COMMAND} --expert_list DoubleTile2DPadAndHoist --problem_sizes_list 192,128,128 --n_iters=200)
  (${COMMAND} --expert_list SingleTiling2DPeel --problem_sizes_list 480,512,16 --n_iters=200)
  (${COMMAND} --expert_list SingleTiling3DPad --problem_sizes_list 384,256,256 --n_iters=200)
  (${COMMAND} --expert_list DoubleTile2DPadAndHoist --problem_sizes_list 480,512,256 --n_iters=200)
  (${COMMAND} --expert_list DoubleTile2DPadAndHoist --problem_sizes_list 784,128,512 --n_iters=200)
  (${COMMAND} --expert_list DoubleTile2DPadAndHoist --problem_sizes_list 1020,1152,1152 --n_iters=200)
  (${COMMAND} --expert_list DoubleTile2DPadAndHoist --problem_sizes_list 1920,2304,2304 --n_iters=200)
}

function matmul_mknk_repro() {
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.matmul.bench --spec_list mk,nk ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list [] "

  (${COMMAND} --expert_list SingleTiling3DPeelTranspose --problem_sizes_list 18,32,96 --n_iters=200)
  (${COMMAND} --expert_list SingleTiling3DPeelTranspose --problem_sizes_list 24,64,96 --n_iters=200)
  (${COMMAND} --expert_list SingleTiling3DPeelTranspose --problem_sizes_list 48,64,128 --n_iters=200)
  (${COMMAND} --expert_list SingleTiling3DPeelTranspose --problem_sizes_list 192,64,128 --n_iters=200)
  (${COMMAND} --expert_list DoubleTile2DPadAndHoist --problem_sizes_list 192,128,128 --n_iters=200)
  (${COMMAND} --expert_list DoubleTile2DPadAndHoist --problem_sizes_list 480,512,16 --n_iters=200)
  (${COMMAND} --expert_list DoubleTile2DPadAndHoist --problem_sizes_list 384,256,256 --n_iters=200)
  (${COMMAND} --expert_list DoubleTile2DPadAndHoist --problem_sizes_list 480,512,256 --n_iters=200)
  (${COMMAND} --expert_list DoubleTile2DPadAndHoist --problem_sizes_list 784,128,512 --n_iters=200)
  (${COMMAND} --expert_list DoubleTile2DPadAndHoist --problem_sizes_list 1020,1152,1152 --n_iters=200)
  (${COMMAND} --expert_list DoubleTile2DPadAndHoist --problem_sizes_list 1920,2304,2304 --n_iters=200)
}

function run_matmul_benchmarks_n_times() {
  if (test -z "$1") || (test -z "$2")
  then
    echo Usage run_matmul_benchmarks_n_times target_benchmark_dir n_times
    exit 1
  fi

  for i in $(seq $2); do
    run_one_and_append_results_to_data matmul_mkkn_repro $1
    run_one_and_append_results_to_data matmul_kmkn_repro $1
    run_one_and_append_results_to_data matmul_mknk_repro $1
  done

  echo To create plots of the results run a command such as: 
  echo python ./tools/plot_benchmark.py \
    --input ${BENCH_DIR}/all.data \
    --output ${BENCH_DIR}/all.pdf \
    --plot_name \"XXXX\" \
    --metric_to_plot \"gbyte_per_s_per_iter\" \
    --benchmarks_to_plot \"matmul_mkkn\"
}