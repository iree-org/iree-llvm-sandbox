
#!/bin/bash

set -ex

# AVX512 throttling needs a lot of iteration, only report the last 100 after
# throttling has had a good chance of happening.
export SANDBOX_KEEP_LAST_N_RUNS=100

export BASE_SCRIPT_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source ${BASE_SCRIPT_PATH}/benchmark.sh

function copy_2d_static_l3_repro() {
  # Passing alwaysinline reduces the variance that is otherwise too high for 
  # small L1 copies.
  export SANDBOX_INLINING='alwaysinline'
  COMMAND="cset proc -s sandbox_0 -e python -- -m python.examples.copy.copy_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"

  (${COMMAND} --expert_list Tile6x32Peel --problem_sizes_list 250,960 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 250,1280 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 250,1920 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 250,2560 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 250,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 800,640 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 800,960 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 800,1280 --n_iters=5000)
  (${COMMAND} --expert_list Tile6x16Peel --problem_sizes_list 800,1920 --n_iters=5000)
  (${COMMAND} --expert_list Tile6x32Peel --problem_sizes_list 800,2560 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 1000,640 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x32Peel --problem_sizes_list 1000,960 --n_iters=5000)
  (${COMMAND} --expert_list Tile6x32Peel --problem_sizes_list 1000,1280 --n_iters=5000)
  (${COMMAND} --expert_list Tile6x32Peel --problem_sizes_list 1000,1920 --n_iters=5000)
  (${COMMAND} --expert_list Tile6x32Peel --problem_sizes_list 1000,2560 --n_iters=5000)
  (${COMMAND} --expert_list Tile6x32Peel --problem_sizes_list 40,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 60,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x16Peel --problem_sizes_list 80,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 100,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 200,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x32Peel --problem_sizes_list 400,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x16Peel --problem_sizes_list 40,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 60,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 80,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 100,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 200,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x32Peel --problem_sizes_list 300,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 2000,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x32Peel --problem_sizes_list 3500,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 5500,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x32Peel --problem_sizes_list 9500,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 20000,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x32Peel --problem_sizes_list 30000,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 40000,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x32Peel --problem_sizes_list 3000,128 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 4500,128 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 6000,128 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x32Peel --problem_sizes_list 9500,128 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16Peel --problem_sizes_list 12000,128 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x32Peel --problem_sizes_list 16000,128 --n_iters=5000)
}


function transpose_2d_static_l3_repro() {
  export SANDBOX_INLINING='alwaysinline'
  COMMAND="cset proc -s sandbox_0 -e python -- -m python.examples.transpose.transpose_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"

  (${COMMAND} --expert_list Tile8x8AVX2 --problem_sizes_list 250,960 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8Shuffle --problem_sizes_list 250,1280 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8AVX2 --problem_sizes_list 250,1920 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8Shuffle --problem_sizes_list 250,2560 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8Shuffle --problem_sizes_list 250,3840 --n_iters=5000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 800,640 --n_iters=5000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 800,960 --n_iters=5000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 800,1280 --n_iters=5000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 800,1920 --n_iters=5000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 800,2560 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8AVX2 --problem_sizes_list 1000,640 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8Shuffle --problem_sizes_list 1000,960 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8Shuffle --problem_sizes_list 1000,1280 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8Shuffle --problem_sizes_list 1000,1920 --n_iters=5000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 1000,2560 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8AVX2 --problem_sizes_list 40,3840 --n_iters=5000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 60,3840 --n_iters=5000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 80,3840 --n_iters=5000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 100,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8Shuffle --problem_sizes_list 200,3840 --n_iters=5000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 400,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8AVX2 --problem_sizes_list 40,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8Shuffle --problem_sizes_list 60,5760 --n_iters=5000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 80,5760 --n_iters=5000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 100,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8Shuffle --problem_sizes_list 200,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8Shuffle --problem_sizes_list 300,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8AVX2 --problem_sizes_list 2000,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8AVX2 --problem_sizes_list 3500,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8AVX2 --problem_sizes_list 5500,64 --n_iters=5000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 9500,64 --n_iters=5000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 20000,64 --n_iters=5000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 30000,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile16x16Shuffle --problem_sizes_list 40000,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8AVX2 --problem_sizes_list 3000,128 --n_iters=5000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 4500,128 --n_iters=5000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 6000,128 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8AVX2 --problem_sizes_list 9500,128 --n_iters=5000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 12000,128 --n_iters=5000)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 16000,128 --n_iters=5000)
}


function row_reduction_2d_static_l3_repro() {
  export SANDBOX_INLINING='alwaysinline'
  COMMAND="cset proc -s sandbox_0 -e python -- -m python.examples.reduction.row_reduction_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"

  (${COMMAND} --expert_list Tile8x64PeelInnerReduction --problem_sizes_list 250,960 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 250,1280 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x64PeelInnerReduction --problem_sizes_list 250,1920 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x64PeelInnerReduction --problem_sizes_list 250,2560 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x64PeelInnerReduction --problem_sizes_list 250,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 800,640 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x64PeelInnerReduction --problem_sizes_list 800,960 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x64PeelInnerReduction --problem_sizes_list 800,1280 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x64PeelInnerReduction --problem_sizes_list 800,1920 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 800,2560 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 1000,640 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x64PeelInnerReduction --problem_sizes_list 1000,960 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x64PeelInnerReduction --problem_sizes_list 1000,1280 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 1000,1920 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x64PeelInnerReduction --problem_sizes_list 1000,2560 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 40,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 60,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 80,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x64PeelInnerReduction --problem_sizes_list 100,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x64PeelInnerReduction --problem_sizes_list 200,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 400,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 40,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 60,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x64PeelInnerReduction --problem_sizes_list 80,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 100,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 200,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x64PeelInnerReduction --problem_sizes_list 300,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile6x128PeelInnerReduction --problem_sizes_list 2000,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x64PeelInnerReduction --problem_sizes_list 3500,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile6x64PeelInnerReduction --problem_sizes_list 5500,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 9500,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 20000,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 30000,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x64PeelInnerReduction --problem_sizes_list 40000,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile6x64PeelInnerReduction --problem_sizes_list 3000,128 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 4500,128 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x128PeelInnerReduction --problem_sizes_list 6000,128 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 9500,128 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 12000,128 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x128PeelInnerReduction --problem_sizes_list 16000,128 --n_iters=5000)
}

function column_reduction_2d_static_l3_repro() {
  export SANDBOX_INLINING='alwaysinline'
  COMMAND="cset proc -s sandbox_0 -e python -- -m python.examples.reduction.column_reduction_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"

  (${COMMAND} --expert_list Tile6x16PeelInnerParallel --problem_sizes_list 250,960 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x16PeelInnerParallel --problem_sizes_list 250,1280 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16PeelInnerParallel --problem_sizes_list 250,1920 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8PeelInnerParallel --problem_sizes_list 250,2560 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8PeelInnerParallel --problem_sizes_list 250,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile6x16PeelInnerParallel --problem_sizes_list 800,640 --n_iters=5000)
  (${COMMAND} --expert_list Tile16x32PeelInnerParallel --problem_sizes_list 800,960 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x16PeelInnerParallel --problem_sizes_list 800,1280 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8PeelInnerParallel --problem_sizes_list 800,1920 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x8PeelInnerParallel --problem_sizes_list 800,2560 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16PeelInnerParallel --problem_sizes_list 1000,640 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8PeelInnerParallel --problem_sizes_list 1000,960 --n_iters=5000)
  (${COMMAND} --expert_list Tile6x8PeelInnerParallel --problem_sizes_list 1000,1280 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8PeelInnerParallel --problem_sizes_list 1000,1920 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8PeelInnerParallel --problem_sizes_list 1000,2560 --n_iters=5000)
  (${COMMAND} --expert_list Tile16x64PeelInnerParallel --problem_sizes_list 40,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x32PeelInnerParallel --problem_sizes_list 60,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x16PeelInnerParallel --problem_sizes_list 80,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x32PeelInnerParallel --problem_sizes_list 100,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x16PeelInnerParallel --problem_sizes_list 200,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile16x32PeelInnerParallel --problem_sizes_list 400,3840 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x16PeelInnerParallel --problem_sizes_list 40,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8PeelInnerParallel --problem_sizes_list 60,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile6x16PeelInnerParallel --problem_sizes_list 80,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile6x16PeelInnerParallel --problem_sizes_list 100,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8PeelInnerParallel --problem_sizes_list 200,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8PeelInnerParallel --problem_sizes_list 300,5760 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x32PeelInnerParallel --problem_sizes_list 2000,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile6x16PeelInnerParallel --problem_sizes_list 3500,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x64PeelInnerParallel --problem_sizes_list 5500,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile6x32PeelInnerParallel --problem_sizes_list 9500,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16PeelInnerParallel --problem_sizes_list 20000,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x16PeelInnerParallel --problem_sizes_list 30000,64 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x32PeelInnerParallel --problem_sizes_list 3000,128 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x64PeelInnerParallel --problem_sizes_list 4500,128 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8PeelInnerParallel --problem_sizes_list 6000,128 --n_iters=5000)
  (${COMMAND} --expert_list Tile4x32PeelInnerParallel --problem_sizes_list 9500,128 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8PeelInnerParallel --problem_sizes_list 12000,128 --n_iters=5000)
  (${COMMAND} --expert_list Tile8x8PeelInnerParallel --problem_sizes_list 16000,128 --n_iters=5000)

  # Bug to investigate!
  # (${COMMAND} --n_iters 1000 --problem_sizes_list 40000,64)
  # Bug to investigate!
  # (${COMMAND} --n_iters 1000 --problem_sizes_list 20000,128)
}

function run_l3_benchmarks_n_times() {
  if (test -z "$1") || (test -z "$2")
  then
    echo Usage run_l3_benchmarks_n_times target_benchmark_dir n_times
    exit 1
  fi

  for i in $(seq $2); do
    run_one_and_append_results_to_data copy_2d_static_l3_repro $1
    run_one_and_append_results_to_data transpose_2d_static_l3_repro $1
    run_one_and_append_results_to_data row_reduction_2d_static_l3_repro $1
    run_one_and_append_results_to_data column_reduction_2d_static_l3_repro $1
  done

  echo To create plots of the results run a command such as: 
  echo python ./tools/plot_benchmark.py \
    --input ${BENCH_DIR}/all.data \
    --output ${BENCH_DIR}/all.pdf \
    --plot_name \"Bandwidth-bound Experiments -- L2-bound\" \
    --metric_to_plot \"gbyte_per_s_per_iter\" \
    --benchmarks_to_plot \"transpose_2d,copy_2d\"
}