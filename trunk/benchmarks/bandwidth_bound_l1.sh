
#!/bin/bash

set -ex

# AVX512 throttling needs a lot of iteration, only report the last 100 after
# throttling has had a good chance of happening.
export SANDBOX_KEEP_LAST_N_RUNS=100

export BASE_SCRIPT_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source ${BASE_SCRIPT_PATH}/benchmark.sh

function copy_2d_static_l1_repro() {
  # Passing alwaysinline reduces the variance that is otherwise too high for 
  # small L1 copies.
  export SANDBOX_INLINING='alwaysinline'
  COMMAND="cset proc -s sandbox_parallel -e python -- -m python.examples.copy.copy_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"

  (${COMMAND} --expert_list Tile8x32Peel --problem_sizes_list 10,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x32Peel --problem_sizes_list 10,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x32Peel --problem_sizes_list 10,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x32Peel --problem_sizes_list 10,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x32Peel --problem_sizes_list 10,128 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x32Peel --problem_sizes_list 16,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x32Peel --problem_sizes_list 16,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x32Peel --problem_sizes_list 16,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 16,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x16Peel --problem_sizes_list 16,128 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x32Peel --problem_sizes_list 20,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x32Peel --problem_sizes_list 20,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 20,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 20,128 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x32Peel --problem_sizes_list 32,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x16Peel --problem_sizes_list 32,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x16Peel --problem_sizes_list 32,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x16Peel --problem_sizes_list 32,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x32Peel --problem_sizes_list 32,128 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 40,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x32Peel --problem_sizes_list 40,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x32Peel --problem_sizes_list 40,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x16Peel --problem_sizes_list 40,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x16Peel --problem_sizes_list 64,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 64,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x32Peel --problem_sizes_list 64,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x32Peel --problem_sizes_list 8,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x32Peel --problem_sizes_list 12,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 20,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x32Peel --problem_sizes_list 8,144 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x16Peel --problem_sizes_list 12,144 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x32Peel --problem_sizes_list 16,144 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x32Peel --problem_sizes_list 20,144 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x16Peel --problem_sizes_list 40,16 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 70,16 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 110,16 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 190,16 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 60,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x32Peel --problem_sizes_list 90,32 --n_iters=10000)
}


function transpose_2d_static_l1_repro() {
  export SANDBOX_INLINING='alwaysinline'
  COMMAND="cset proc -s sandbox_parallel -e python -- -m python.examples.transpose.transpose_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"

  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 10,32 --n_iters=10000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 10,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 10,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 10,96 --n_iters=10000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 10,128 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 16,32 --n_iters=10000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 16,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 16,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 16,128 --n_iters=10000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 20,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 20,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 20,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 20,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 20,128 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 32,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 32,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 32,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 32,96 --n_iters=10000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 32,128 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 40,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 40,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 40,64 --n_iters=10000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 40,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 64,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 64,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 64,64 --n_iters=10000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 8,96 --n_iters=10000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 12,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 16,96 --n_iters=10000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 8,144 --n_iters=10000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 12,144 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 16,144 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 20,144 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 40,16 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 70,16 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 110,16 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 190,16 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 60,32 --n_iters=10000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 90,32 --n_iters=10000)
}

function row_reduction_2d_static_l1_repro() {
  export SANDBOX_INLINING='alwaysinline'
  COMMAND="cset proc -s sandbox_parallel -e python -- -m python.examples.reduction.row_reduction_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"

  (${COMMAND} --expert_list Tile4x64PeelInnerReduction --problem_sizes_list 10,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x64PeelInnerReduction --problem_sizes_list 10,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x128PeelInnerReduction --problem_sizes_list 10,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x128PeelInnerReduction --problem_sizes_list 10,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x128PeelInnerReduction --problem_sizes_list 10,128 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x128PeelInnerReduction --problem_sizes_list 16,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x128PeelInnerReduction --problem_sizes_list 16,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x64PeelInnerReduction --problem_sizes_list 16,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x128PeelInnerReduction --problem_sizes_list 16,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 16,128 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x64PeelInnerReduction --problem_sizes_list 20,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x128PeelInnerReduction --problem_sizes_list 20,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x128PeelInnerReduction --problem_sizes_list 20,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 20,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 20,128 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x64PeelInnerReduction --problem_sizes_list 32,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x64PeelInnerReduction --problem_sizes_list 32,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x64PeelInnerReduction --problem_sizes_list 32,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 32,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 32,128 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x128PeelInnerReduction --problem_sizes_list 40,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x64PeelInnerReduction --problem_sizes_list 40,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x128PeelInnerReduction --problem_sizes_list 40,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x128PeelInnerReduction --problem_sizes_list 40,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x128PeelInnerReduction --problem_sizes_list 64,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x64PeelInnerReduction --problem_sizes_list 64,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x64PeelInnerReduction --problem_sizes_list 64,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x128PeelInnerReduction --problem_sizes_list 8,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 12,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x128PeelInnerReduction --problem_sizes_list 8,144 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x128PeelInnerReduction --problem_sizes_list 12,144 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 16,144 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 20,144 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x64PeelInnerReduction --problem_sizes_list 40,16 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x64PeelInnerReduction --problem_sizes_list 70,16 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x16PeelInnerReduction --problem_sizes_list 110,16 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x128PeelInnerReduction --problem_sizes_list 190,16 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x128PeelInnerReduction --problem_sizes_list 60,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x128PeelInnerReduction --problem_sizes_list 90,32 --n_iters=10000)
}

function column_reduction_2d_static_l1_repro() {
  export SANDBOX_INLINING='alwaysinline'
  COMMAND="cset proc -s sandbox_parallel -e python -- -m python.examples.reduction.column_reduction_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  
  (${COMMAND} --expert_list Tile4x16PeelInnerParallel --problem_sizes_list 10,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x16PeelInnerParallel --problem_sizes_list 10,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x64PeelInnerParallel --problem_sizes_list 10,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x32PeelInnerParallel --problem_sizes_list 10,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x32PeelInnerParallel --problem_sizes_list 10,128 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x16PeelInnerParallel --problem_sizes_list 16,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x64PeelInnerParallel --problem_sizes_list 16,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x32PeelInnerParallel --problem_sizes_list 16,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x64PeelInnerParallel --problem_sizes_list 16,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x64PeelInnerParallel --problem_sizes_list 16,128 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x16PeelInnerParallel --problem_sizes_list 20,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x32PeelInnerParallel --problem_sizes_list 20,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x64PeelInnerParallel --problem_sizes_list 20,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x64PeelInnerParallel --problem_sizes_list 20,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x64PeelInnerParallel --problem_sizes_list 20,128 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x32PeelInnerParallel --problem_sizes_list 32,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x64PeelInnerParallel --problem_sizes_list 32,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x32PeelInnerParallel --problem_sizes_list 32,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x64PeelInnerParallel --problem_sizes_list 32,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x64PeelInnerParallel --problem_sizes_list 32,128 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x32PeelInnerParallel --problem_sizes_list 40,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x32PeelInnerParallel --problem_sizes_list 40,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x32PeelInnerParallel --problem_sizes_list 40,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x64PeelInnerParallel --problem_sizes_list 64,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x64PeelInnerParallel --problem_sizes_list 64,48 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x64PeelInnerParallel --problem_sizes_list 64,64 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x32PeelInnerParallel --problem_sizes_list 8,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x64PeelInnerParallel --problem_sizes_list 12,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x32PeelInnerParallel --problem_sizes_list 40,96 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x32PeelInnerParallel --problem_sizes_list 8,144 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x64PeelInnerParallel --problem_sizes_list 12,144 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x32PeelInnerParallel --problem_sizes_list 16,144 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x16PeelInnerParallel --problem_sizes_list 20,144 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x16PeelInnerParallel --problem_sizes_list 40,16 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x32PeelInnerParallel --problem_sizes_list 70,16 --n_iters=10000)
  (${COMMAND} --expert_list Tile8x32PeelInnerParallel --problem_sizes_list 110,16 --n_iters=10000)
  (${COMMAND} --expert_list Tile6x16PeelInnerParallel --problem_sizes_list 190,16 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x32PeelInnerParallel --problem_sizes_list 60,32 --n_iters=10000)
  (${COMMAND} --expert_list Tile4x16PeelInnerParallel --problem_sizes_list 90,32 --n_iters=10000)
}

function run_l1_benchmarks_n_times() {
  if (test -z "$1") || (test -z "$2")
  then
    echo Usage run_l1_benchmarks_n_times target_benchmark_dir n_times
    exit 1
  fi

  for i in $(seq $2); do
    run_one_and_append_results_to_data copy_2d_static_l1_repro $1
    run_one_and_append_results_to_data transpose_2d_static_l1_repro $1
    run_one_and_append_results_to_data row_reduction_2d_static_l1_repro $1
    run_one_and_append_results_to_data column_reduction_2d_static_l1_repro $1
  done

  echo To create plots of the results run a command such as: 
  echo python ./tools/plot_benchmark.py \
    --input ${BENCH_DIR}/all.data \
    --output ${BENCH_DIR}/all.pdf \
    --plot_name \"Bandwidth-bound Experiments -- L1-bound\" \
    --metric_to_plot \"gbyte_per_s_per_iter\" \
    --benchmarks_to_plot \"transpose_2d,copy_2d\"
}