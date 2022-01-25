#!/bin/bash

set -ex

###############################################################################
# Ran on a machine with the following characteristics.
# Note the SMT pair of CPU 4 (i.e. CPU 40) is disabled.
###############################################################################
# > lscpu
# Architecture:            x86_64
#   CPU op-mode(s):        32-bit, 64-bit
#   Address sizes:         46 bits physical, 48 bits virtual
#   Byte Order:            Little Endian
# CPU(s):                  72
#   On-line CPU(s) list:   0-39,41-71
#   Off-line CPU(s) list:  40
# Vendor ID:               GenuineIntel
#   BIOS Vendor ID:        Intel(R) Corporation
#   Model name:            Intel(R) Xeon(R) Gold 6154 CPU @ 3.00GHz
#     BIOS Model name:     Intel(R) Xeon(R) Gold 6154 CPU @ 3.00GHz
#     CPU family:          6
#     Model:               85
#     Thread(s) per core:  2
#     Core(s) per socket:  18
#     Socket(s):           2
#     Stepping:            4
#     CPU max MHz:         3700.0000
#     CPU min MHz:         1200.0000
#     BogoMIPS:            6000.00
#     Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr p
#                          dcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cdp_l3 invpcid_single pti intel_ppin ssbd mba ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm cqm mpx rd
#                          t_a avx512f avx512dq rdseed adx smap clflushopt clwb intel_pt avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req pku ospke md_clear flush_l1d
# Virtualization features: 
#   Virtualization:        VT-x
# Caches (sum of all):     
#   L1d:                   1.1 MiB (36 instances)
#   L1i:                   1.1 MiB (36 instances)
#   L2:                    36 MiB (36 instances)
#   L3:                    49.5 MiB (2 instances)
# NUMA:                    
#   NUMA node(s):          2
#   NUMA node0 CPU(s):     0-17,36-39,41-53
#   NUMA node1 CPU(s):     18-35,54-71
# Vulnerabilities:        

###############################################################################
# The benchmarks below assume the setup described in the 'Benchmark commands' 
# section in the README.md. Instructions are reproduced here for convenience.
###############################################################################
# ################################################################
# # Prepare to run on CPU 4 only
# ################################################################
# # Disable address space randomization.
# echo 0 > /proc/sys/kernel/randomize_va_space

# # Disable the sibling of CPU 4.
# cat /sys/devices/system/cpu/cpu4/topology/thread_siblings_list 
# # E.g. this may return 4,40
# echo 0 > /sys/devices/system/cpu/cpu40/online

# ################################################################
# Perform cpuset manipulation.
# ################################################################
# # For reference, cset shield does not seem to run as expected on at least 2 systems.
# # cset shield -c 4 --user=${RUN_AS_USER} -k on --userset=${RUN_AS_USER}
# # Instead, reproduce the follwing: 
# # https://documentation.suse.com/sle-rt/15-SP2/html/SLE-RT-all/cha-shielding-cpuset.html
# #
# cset set -c 0-3,5-39,41-71 -s system -s system
# cset set -s sandbox -c 4 -m 0 --cpu_exclusive
# cset proc -m -f root -t system

# ################################################################
# # Freq control (note, cloud VM instances do not allow).
# ################################################################

# echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo
# echo performance > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
###############################################################################

export IREE_LLVM_SANDBOX_BUILD_DIR=$(dirname $0)/build 
export MLIR_RUNNER_UTILS_LIB=${IREE_LLVM_SANDBOX_BUILD_DIR}/lib/libmlir_runner_utils.so 
export MLIR_C_RUNNER_UTILS_LIB=${IREE_LLVM_SANDBOX_BUILD_DIR}/lib/libmlir_c_runner_utils.so 
export PYTHONPATH=${PYTHONPATH}:${IREE_LLVM_SANDBOX_BUILD_DIR}/tools/sandbox/python_package 
export PATH=$PATH:$(dirname ~ntv)/.venv/mlirdev/bin/

function prepare_data_collection() {
  DUMP_DATA_FLAG=""
  PLOT_COMMAND_LINE=""
  if !(test -z "$1" ) 
  then
    DUMP_DATA_FLAG="--dump_data $1"
  fi
}

###############################################################################
# Copy benchmarks.
###############################################################################
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
# [15000, 272],  # 120% L3 load, L3 BW @ 15.8GB/s
#
# [30000, 272], # DRAM (2.4x L3 load), L3 BW @ 12.2GB/s
# [300000, 272], # DRAM (24x L3 load), L3 BW @ 10.8GB/s
function copy_bandwidth_benchmark() {
  cset proc -s sandbox -e python -- -m python.examples.copy.custom_copy_2d_bench
}

###############################################################################
# Static 1D copy benchmarks.
###############################################################################
# Careful here, static problem size smaller than the tile sizes completely folds
# away since the result tensor is not used.
# TODO: add a fake noop use after the timer in the timing loop to avoid this.
function copy_1d_static_small() {
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.copy.copy_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 32)
  (${COMMAND} --problem_sizes_list 128)
  (${COMMAND} --problem_sizes_list 1024)
  (${COMMAND} --problem_sizes_list 2000)
  (${COMMAND} --problem_sizes_list 2040)
  (${COMMAND} --problem_sizes_list 4000)
  (${COMMAND} --problem_sizes_list 4096)
}

###############################################################################
# Static 2D copy benchmarks.
###############################################################################
# Careful here, static problem size smaller than the tile sizes completely folds
# away since the result tensor is not used.
# TODO: add a fake noop use after the timer in the timing loop to avoid this.
function copy_2d_static_small_repro() {
  # Passing alwaysinline reduces the variance that is otherwise too high for 
  # small L1 copies.
  export SANDBOX_INLINING='alwaysinline'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.copy.copy_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  ${COMMAND} --problem_sizes_list 1,272
  (${COMMAND} --problem_sizes_list 2,272)
  (${COMMAND} --problem_sizes_list 3,272)
  (${COMMAND} --problem_sizes_list 4,272)
  (${COMMAND} --problem_sizes_list 5,272)
  (${COMMAND} --problem_sizes_list 6,272)
  (${COMMAND} --problem_sizes_list 7,272)
  (${COMMAND} --problem_sizes_list 8,272)
  (${COMMAND} --problem_sizes_list 9,272)
  (${COMMAND} --problem_sizes_list 10,272)
  (${COMMAND} --problem_sizes_list 11,272)
  (${COMMAND} --problem_sizes_list 12,272)
  (${COMMAND} --problem_sizes_list 13,272)
  (${COMMAND} --problem_sizes_list 14,272)
}
###############################################################################
# Static 2D transpose benchmarks.
###############################################################################
function transpose_2d_static_small_repro_median() {
  export SANDBOX_INLINING='alwaysinline'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.transpose.transpose_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list Tile8x8Shuffle --problem_sizes_list 16,16)
  (${COMMAND} --expert_list Tile16x16Shuffle --problem_sizes_list 16,16)
  (${COMMAND} --expert_list Tile8x8AVX2 --problem_sizes_list 16,16)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 16,16)

  (${COMMAND} --expert_list Tile8x8Shuffle --problem_sizes_list 32,32)
  (${COMMAND} --expert_list Tile16x16Shuffle --problem_sizes_list 32,32)
  (${COMMAND} --expert_list Tile8x8AVX2 --problem_sizes_list 32,32)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 32,32)
}

function transpose_2d_static_l1_repro() {
  export SANDBOX_INLINING='alwaysinline'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.transpose.transpose_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
   # 512B R + 512B W -> @3.125% L1
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 32,4)
   # 2048B R + 2048B W -> @6.25% L1
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 32,8)
   # 2048B R + 2048B W -> @12.5% L1
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 32,16)
   # 4096B R + 4096B W -> @25% L1
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 32,32)
  #(${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 32,48)
   # 8192B R + 8192B W -> @50% L1
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 32,64)
  #(${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 32,80)
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 32,96)
  #(${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 32,112)
   # 16KB R + 16KB W -> @100% L1
  (${COMMAND} --expert_list Tile4x8Shuffle --problem_sizes_list 32,128)
}

function transpose_2d_static_l2_repro() {
  export SANDBOX_INLINING='alwaysinline'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.transpose.transpose_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"

   # 32KB R + 32KB W -> @6.4% L2
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 32,256)
  #(${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 64,128)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 128,64)
  
   # 64KB R + 64KB W -> @12.8% L2
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 32,512)
  #(${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 64,256)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 128,128)
  
   # 128KB R + 128KB W -> @25.6% L2
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 32,1024)
  #(${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 64,512)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 128,256)

   # 128KB R + 128KB W -> @51.0% L2
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 32,2048)
  #(${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 64,1024)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 128,512)

   # 128KB R + 128KB W -> @102.0% L2
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 32,4096)
  #(${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 64,2048)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 128,1024)
}

function transpose_2d_static_l3_repro() {
  export SANDBOX_INLINING='alwaysinline'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.transpose.transpose_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  # 2MB R + 2MB W -> @16.0% L3
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 32,16384)
  #(${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 64,8192)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 128,4096)
  #(${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 256,2048)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 512,1024)

  # 4MB R + 4MB W -> @32.0% L3
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 32,32768)
  #(${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 64,16384)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 128,8192)
  #(${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 256,4096)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 512,2048)

  # 8MB R + 8MB W -> @64.0% L3
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 32,65536)
  #(${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 64,32768)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 128,16384)
  #(${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 256,8192)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 512,4096)

  # 12MB R + 12MB W -> @96.0% L3
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 48,65536)
  #(${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 96,32768)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 192,16384)
  #(${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 384,8192)
  (${COMMAND} --expert_list TripleTile4x8Shuffle --problem_sizes_list 768,4096)
}

###############################################################################
# Static 1D reduction benchmarks.
###############################################################################
function reduction_1d_static_l1_repro() {
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.reduction.reduction_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 100)
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 1000)
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 2048)
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 3333)
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 4567)
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 8000)
}

function reduction_1d_static_l2_repro() {
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.reduction.reduction_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 10000)
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 20000)
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 50000)
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 100000)
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 200000)
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 250000)
}

function reduction_1d_static_l3_repro() {
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.reduction.reduction_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 1000000)
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 2000000)
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 3000000)
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 4000000)
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 5000000)
}

function reduction_1d_static_dram_repro() {
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.reduction.reduction_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 10000000)
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 20000000)
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 30000000)
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 40000000)
  (${COMMAND} --expert_list Tile1DPeel --problem_sizes_list 50000000)
}

###############################################################################
# Static 2D row reduction benchmarks.
###############################################################################
function row_reduction_2d_static_l1_repro() {
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.reduction.row_reduction_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 50,47)
  (${COMMAND} --problem_sizes_list 99,88)
  (${COMMAND} --problem_sizes_list 100,256)
  (${COMMAND} --problem_sizes_list 125,347)
  (${COMMAND} --problem_sizes_list 200,384)
}
function row_reduction_2d_static_l2_repro() {
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.reduction.row_reduction_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 200,512)
  (${COMMAND} --problem_sizes_list 250,547)
  (${COMMAND} --problem_sizes_list 250,1000)
  (${COMMAND} --problem_sizes_list 300,866)
}
function row_reduction_2d_static_l3_repro() {
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.reduction.row_reduction_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1000,1024)
  (${COMMAND} --problem_sizes_list 2000,2024)
  (${COMMAND} --problem_sizes_list 2000,3024)
  (${COMMAND} --problem_sizes_list 3000,2024)
}

###############################################################################
# Static 2D column reduction benchmarks.
###############################################################################
function column_reduction_2d_static_l1_repro() {
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.reduction.column_reduction_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list Tile8x64PeelInnerParallel --problem_sizes_list 50,47)
  (${COMMAND} --expert_list Tile16x32PeelInnerParallel --problem_sizes_list 50,100)
  (${COMMAND} --expert_list Tile16x32PeelInnerParallel --problem_sizes_list 60,102)
}
function column_reduction_2d_static_l2_repro() {
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.reduction.column_reduction_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list Tile16x32PeelInnerParallel --problem_sizes_list 99,88)
  (${COMMAND} --expert_list Tile16x32PeelInnerParallel --problem_sizes_list 100,256)
  (${COMMAND} --expert_list Tile16x32PeelInnerParallel --problem_sizes_list 125,347)
  (${COMMAND} --expert_list Tile16x32PeelInnerParallel --problem_sizes_list 200,384)
}
function column_reduction_2d_static_l3_repro() {
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.reduction.column_reduction_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list Tile8x64PeelInnerParallel --problem_sizes_list 1000,1024)
  (${COMMAND} --expert_list Tile8x64PeelInnerParallel --problem_sizes_list 2000,2024)
  (${COMMAND} --expert_list Tile8x64PeelInnerParallel --problem_sizes_list 2000,3024)
  (${COMMAND} --expert_list Tile8x64PeelInnerParallel --problem_sizes_list 3000,2024)
}

###############################################################################
# Static matmul mk,kn benchmarks.
###############################################################################
function matmul_static_small() {
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.matmul.bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling2DPeel  --spec_list mk,kn --problem_sizes_list 18,32,96 )
  (${COMMAND} --expert_list SingleTiling2DPeel  --spec_list mk,kn --problem_sizes_list 24,64,96 )
  (${COMMAND} --expert_list SingleTiling3DPeel  --spec_list mk,kn --problem_sizes_list 48,64,128 )
}

function matmul_static_small_reduction_dimension() {
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.matmul.bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling2DPeel  --spec_list mk,kn --problem_sizes_list 480,512,16 )
}

function matmul_static_medium() {
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.matmul.bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPad  --spec_list mk,kn --problem_sizes_list  384,256,256 )
  (${COMMAND} --expert_list SingleTiling3DPad  --spec_list mk,kn --problem_sizes_list  480,512,256 )
  (${COMMAND} --expert_list SingleTiling2DPeel  --spec_list mk,kn --problem_sizes_list  784,128,512 )
}

function matmul_static_large() {
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.matmul.bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list DoubleTile2DPadAndHoist  --spec_list mk,kn --problem_sizes_list  1020,1152,1152 )
  (${COMMAND} --expert_list DoubleTile2DPadAndHoist  --spec_list mk,kn --problem_sizes_list  1920,2304,2304 )
  (${COMMAND} --expert_list DoubleTile2DPadAndHoist  --spec_list mk,kn --problem_sizes_list  2304,2304,2560 )
}

###############################################################################
# Static conv1d nwc benchmarks.
###############################################################################
# Batch size 1, 32 -> 64 channels, stride 1, dilation 1.
function conv_1d_static_small_stride_1_dilation_1() {
  STRIDES_AND_DILATIONS='[1],[1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,256,32,3,64,${STRIDES_AND_DILATIONS})
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,988,32,3,64,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,4144,32,3,64,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,11300,32,3,64,${STRIDES_AND_DILATIONS} )
}
# Batch size 1, 32 -> 64 channels, stride 2, dilation 1.
function conv_1d_static_small_stride_2_dilation_1() {
  STRIDES_AND_DILATIONS='[2],[1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,256,32,3,64,${STRIDES_AND_DILATIONS})
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,988,32,3,64,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,4144,32,3,64,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,11300,32,3,64,${STRIDES_AND_DILATIONS} )
}
# Batch size 1, 32 -> 64 channels, stride 1, dilation 2.
function conv_1d_static_small_stride_1_dilation_2() {
  STRIDES_AND_DILATIONS='[1],[2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,256,32,3,64,${STRIDES_AND_DILATIONS})
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,988,32,3,64,${STRIDES_AND_DILATIONS})
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,4144,32,3,64,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,11300,32,3,64,${STRIDES_AND_DILATIONS} )
}
# Batch size 1, 32 -> 64 channels, stride 2, dilation 2.
function conv_1d_static_small_stride_2_dilation_2() {
  STRIDES_AND_DILATIONS='[2],[2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,256,32,3,64,${STRIDES_AND_DILATIONS})
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,988,32,3,64,${STRIDES_AND_DILATIONS})
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,4144,32,3,64,[2],[2])
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,11300,32,3,64,${STRIDES_AND_DILATIONS} )
}

# Batch size 1, 128 -> 256 channels, stride 1, dilation 1.
function conv_1d_static_medium_stride_1_dilation_1() {
  STRIDES_AND_DILATIONS='[1],[1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,256,128,3,256,${STRIDES_AND_DILATIONS})
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,988,128,3,256,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPad  --problem_sizes_list 1,4144,128,3,256,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,11300,128,3,256,${STRIDES_AND_DILATIONS} )
}
# Batch size 1, 128 -> 256 channels, stride 2, dilation 1.
function conv_1d_static_medium_stride_2_dilation_1() {
  STRIDES_AND_DILATIONS='[2],[1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,256,128,3,256,${STRIDES_AND_DILATIONS})
  (${COMMAND} --expert_list SingleTiling3DPad  --problem_sizes_list 1,988,128,3,256,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,4144,128,3,256,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPad  --problem_sizes_list 1,11300,128,3,256,${STRIDES_AND_DILATIONS} )
}
# Batch size 1, 128 -> 256 channels, stride 1, dilation 2.
function conv_1d_static_medium_stride_1_dilation_2() {
  STRIDES_AND_DILATIONS='[1],[2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,256,128,3,256,${STRIDES_AND_DILATIONS})
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,988,128,3,256,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,4144,128,3,256,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,11300,128,3,256,${STRIDES_AND_DILATIONS} )
}
# Batch size 1, 128 -> 256 channels, stride 2, dilation 2.
function conv_1d_static_medium_stride_2_dilation_2() {
  STRIDES_AND_DILATIONS='[2],[2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,256,128,3,256,${STRIDES_AND_DILATIONS})
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,988,128,3,256,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,4144,128,3,256,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,11300,128,3,256,${STRIDES_AND_DILATIONS} )
}

# Batch size 1, 512 -> 1024 channels, stride 1, dilation 1.
function conv_1d_static_large_stride_1_dilation_1() {
  STRIDES_AND_DILATIONS='[1],[1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list DoubleTile3DPeel  --problem_sizes_list 1,256,512,3,1024,${STRIDES_AND_DILATIONS})
  (${COMMAND} --expert_list DoubleTile3DPad  --problem_sizes_list 1,988,512,3,1024,${STRIDES_AND_DILATIONS})
  (${COMMAND} --expert_list DoubleTile3DPad  --problem_sizes_list 1,4144,512,3,1024,${STRIDES_AND_DILATIONS})
  (${COMMAND} --expert_list DoubleTile3DPad  --problem_sizes_list 1,11300,512,3,1024,${STRIDES_AND_DILATIONS} )
}
# Batch size 1, 512 -> 1024 channels, stride 2, dilation 1.
function conv_1d_static_large_stride_2_dilation_1() {
  STRIDES_AND_DILATIONS='[2],[1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list DoubleTile3DPeel  --problem_sizes_list 1,256,512,3,1024,${STRIDES_AND_DILATIONS})
  (${COMMAND} --expert_list  DoubleTile3DPad  --problem_sizes_list 1,988,512,3,1024,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list  DoubleTile3DPad  --problem_sizes_list 1,4144,512,3,1024,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list  DoubleTile3DPad  --problem_sizes_list 1,11300,512,3,1024,${STRIDES_AND_DILATIONS} )
}
# Batch size 1, 512 -> 1024 channels, stride 1, dilation 2.
function conv_1d_static_large_stride_1_dilation_2() {
  STRIDES_AND_DILATIONS='[1],[2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list DoubleTile3DPeel  --problem_sizes_list 1,256,512,3,1024,${STRIDES_AND_DILATIONS})
  (${COMMAND} --expert_list  DoubleTile3DPad  --problem_sizes_list 1,988,512,3,1024,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list  DoubleTile3DPad  --problem_sizes_list 1,4144,512,3,1024,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list  DoubleTile3DPad  --problem_sizes_list 1,11300,512,3,1024,${STRIDES_AND_DILATIONS} )
}
# Batch size 1, 512 -> 1024 channels, stride 1, dilation 2.
function conv_1d_static_large_stride_2_dilation_2() {
  STRIDES_AND_DILATIONS='[2],[2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list DoubleTile3DPeel  --problem_sizes_list 1,256,512,3,1024,${STRIDES_AND_DILATIONS})
  (${COMMAND} --expert_list  DoubleTile3DPad  --problem_sizes_list 1,988,512,3,1024,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list  DoubleTile3DPad  --problem_sizes_list 1,4144,512,3,1024,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list  DoubleTile3DPad  --problem_sizes_list 1,11300,512,3,1024,${STRIDES_AND_DILATIONS} )
}

###############################################################################
# Static conv2d nhwc benchmarks.
###############################################################################
# Batch size 1, 32 -> 64 channels, stride [1, 1], dilation [1, 1].
function conv_2d_static_small_stride_1_dilation_1() {
  STRIDES_AND_DILATIONS='[1,1],[1,1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,16,16,32,3,3,64,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,26,38,32,3,3,64,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,56,74,32,3,3,64,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,100,113,32,3,3,64,${STRIDES_AND_DILATIONS} )
}
# Batch size 1, 32 -> 64 channels, stride [2, 2], dilation [1, 1].
function conv_2d_static_small_stride_2_dilation_1() {
  STRIDES_AND_DILATIONS='[2,2],[1,1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,16,16,32,3,3,64,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,26,38,32,3,3,64,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,56,74,32,3,3,64,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,100,113,32,3,3,64,${STRIDES_AND_DILATIONS} )
}
# Batch size 1, 32 -> 64 channels, stride [1, 1], dilation [2, 2].
function conv_2d_static_small_stride_1_dilation_2() {
  STRIDES_AND_DILATIONS='[1,1],[2,2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,16,16,32,3,3,64,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,26,38,32,3,3,64,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,56,74,32,3,3,64,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,100,113,32,3,3,64,${STRIDES_AND_DILATIONS} )
}
# Batch size 1, 32 -> 64 channels, stride [2, 2], dilation [2, 2].
function conv_2d_static_small_stride_2_dilation_2() {
  STRIDES_AND_DILATIONS='[2,2],[2,2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,16,16,32,3,3,64,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,26,38,32,3,3,64,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,56,74,32,3,3,64,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,100,113,32,3,3,64,${STRIDES_AND_DILATIONS} )
}

# Batch size 1, 128 -> 256 channels, stride [1, 1], dilation [1, 1].
function conv_2d_static_medium_stride_1_dilation_1() {
  STRIDES_AND_DILATIONS='[1,1],[1,1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,16,16,128,3,3,256,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,26,38,128,3,3,256,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,56,74,128,3,3,256,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,100,113,128,3,3,256,${STRIDES_AND_DILATIONS} )
}
# Batch size 1, 128 -> 256 channels, stride [2, 2], dilation [1, 1].
function conv_2d_static_medium_stride_2_dilation_1() {
  STRIDES_AND_DILATIONS='[2,2],[1,1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,16,16,128,3,3,256,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,26,38,128,3,3,256,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,56,74,128,3,3,256,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,100,113,128,3,3,256,${STRIDES_AND_DILATIONS} )
}
# Batch size 1, 128 -> 256 channels, stride [1, 1], dilation [2, 2].
function conv_2d_static_medium_stride_1_dilation_2() {
  STRIDES_AND_DILATIONS='[1,1],[2,2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,16,16,128,3,3,256,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,26,38,128,3,3,256,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,56,74,128,3,3,256,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,100,113,128,3,3,256,${STRIDES_AND_DILATIONS} )
}
# Batch size 1, 128 -> 256 channels, stride [2, 2], dilation [2, 2].
function conv_2d_static_medium_stride_2_dilation_2() {
  STRIDES_AND_DILATIONS='[2,2],[2,2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,16,16,128,3,3,256,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,26,38,128,3,3,256,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,56,74,128,3,3,256,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,100,113,128,3,3,256,${STRIDES_AND_DILATIONS} )
}

# Batch size 1, 512 -> 1024 channels, stride [1, 1], dilation [1, 1].
function conv_2d_static_large_stride_1_dilation_1() {
  STRIDES_AND_DILATIONS='[1,1],[1,1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,16,16,512,3,3,1024,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,26,38,512,3,3,1024,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,56,74,512,3,3,1024,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,100,113,512,3,3,1024,${STRIDES_AND_DILATIONS} )
}
# Batch size 1, 512 -> 1024 channels, stride [2, 2], dilation [1, 1].
function conv_2d_static_large_stride_2_dilation_1() {
  STRIDES_AND_DILATIONS='[2,2],[1,1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,16,16,512,3,3,1024,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,26,38,512,3,3,1024,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,56,74,512,3,3,1024,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,100,113,512,3,3,1024,${STRIDES_AND_DILATIONS} )
}
# Batch size 1, 512 -> 1024 channels, stride [1, 1], dilation [2, 2].
function conv_2d_static_large_stride_1_dilation_2() {
  STRIDES_AND_DILATIONS='[1,1],[2,2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,16,16,512,3,3,1024,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,26,38,512,3,3,1024,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,56,74,512,3,3,1024,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,100,113,512,3,3,1024,${STRIDES_AND_DILATIONS} )
}
# Batch size 1, 512 -> 1024 channels, stride [2, 2], dilation [2, 2].
function conv_2d_static_large_stride_2_dilation_2() {
  STRIDES_AND_DILATIONS='[2,2],[2,2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.conv.conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,16,16,512,3,3,1024,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,26,38,512,3,3,1024,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,56,74,512,3,3,1024,${STRIDES_AND_DILATIONS} )
  (${COMMAND} --expert_list SingleTiling3DPeel  --problem_sizes_list 1,100,113,512,3,3,1024,${STRIDES_AND_DILATIONS} )

}
###############################################################################
# Static depthwise_conv_1d nwc benchmarks.
###############################################################################
# Batch size 1, 32 channels, stride 1, dilation 1.
function depthwise_conv_1d_static_small_stride_1_dilation_1() {
  STRIDES_AND_DILATIONS='[1],[1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,256,32,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,988,32,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,4144,32,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,11300,32,3,${STRIDES_AND_DILATIONS})
}
# Batch size 1, 32 channels, stride 2, dilation 1.
function depthwise_conv_1d_static_small_stride_2_dilation_1() {
  STRIDES_AND_DILATIONS='[2],[1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,256,32,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,988,32,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,4144,32,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,11300,32,3,${STRIDES_AND_DILATIONS})
}
# Batch size 1, 32 channels, stride 1, dilation 2.
function depthwise_conv_1d_static_small_stride_1_dilation_2() {
  STRIDES_AND_DILATIONS='[1],[2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,256,32,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,988,32,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,4144,32,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,11300,32,3,${STRIDES_AND_DILATIONS})
}
# Batch size 1, 32 channels, stride 2, dilation 2.
function depthwise_conv_1d_static_small_stride_2_dilation_2() {
  STRIDES_AND_DILATIONS='[2],[2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,256,32,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,988,32,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,4144,32,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,11300,32,3,${STRIDES_AND_DILATIONS})
}

# Batch size 1, 128 channels, stride 1, dilation 1.
function depthwise_conv_1d_static_medium_stride_1_dilation_1() {
  STRIDES_AND_DILATIONS='[1],[1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,256,128,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,988,128,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,4144,128,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,11300,128,3,${STRIDES_AND_DILATIONS})
}
# Batch size 1, 128 channels, stride 2, dilation 1.
function depthwise_conv_1d_static_medium_stride_2_dilation_1() {
  STRIDES_AND_DILATIONS='[2],[1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,256,128,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,988,128,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,4144,128,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,11300,128,3,${STRIDES_AND_DILATIONS})
}
# Batch size 1, 128 channels, stride 1, dilation 2.
function depthwise_conv_1d_static_medium_stride_1_dilation_2() {
  STRIDES_AND_DILATIONS='[1],[2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,256,128,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,988,128,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,4144,128,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,11300,128,3,${STRIDES_AND_DILATIONS})
}
# Batch size 1, 128 channels, stride 1, dilation 2.
function depthwise_conv_1d_static_medium_stride_2_dilation_2() {
  STRIDES_AND_DILATIONS='[2],[2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,256,128,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,988,128,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,4144,128,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,11300,128,3,${STRIDES_AND_DILATIONS})
}

# Batch size 1, 1024 channels, stride 1, dilation 1.
function depthwise_conv_1d_static_large_stride_1_dilation_1() {
  STRIDES_AND_DILATIONS='[1],[1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,256,1024,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,988,1024,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,4144,1024,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,11300,1024,3,${STRIDES_AND_DILATIONS})
}
# Batch size 1, 1024 channels, stride 2, dilation 1.
function depthwise_conv_1d_static_large_stride_2_dilation_1() {
  STRIDES_AND_DILATIONS='[2],[1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,256,1024,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,988,1024,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,4144,1024,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,11300,1024,3,${STRIDES_AND_DILATIONS})
}
# Batch size 1, 1024 channels, stride 1, dilation 2.
function depthwise_conv_1d_static_large_stride_1_dilation_2() {
  STRIDES_AND_DILATIONS='[1],[2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,256,1024,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,988,1024,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,4144,1024,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,11300,1024,3,${STRIDES_AND_DILATIONS})
}
# Batch size 1, 1024 channels, stride 1, dilation 2.
function depthwise_conv_1d_static_large_stride_2_dilation_2() {
  STRIDES_AND_DILATIONS='[2],[2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_1d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,256,1024,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,988,1024,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,4144,1024,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,11300,1024,3,${STRIDES_AND_DILATIONS})
}

###############################################################################
# Static depthwise_conv_2d nhwc benchmarks.
###############################################################################
# Batch size 1, 32 channels, stride 1, dilation 1.
function depthwise_conv_2d_static_small_stride_1_dilation_1() {
  STRIDES_AND_DILATIONS='[1,1],[1,1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,16,16,32,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,26,38,32,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,56,74,32,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,100,113,32,3,3,${STRIDES_AND_DILATIONS} )
}
# Batch size 1, 32 channels, stride 2, dilation 1.
function depthwise_conv_2d_static_small_stride_2_dilation_1() {
  STRIDES_AND_DILATIONS='[2,2],[1,1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,16,16,32,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,26,38,32,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,56,74,32,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,100,113,32,3,3,${STRIDES_AND_DILATIONS})
}
# Batch size 1, 32 channels, stride 1, dilation 2.
function depthwise_conv_2d_static_small_stride_1_dilation_2() {
  STRIDES_AND_DILATIONS='[1,1],[2,2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,16,16,32,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,26,38,32,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,56,74,32,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,100,113,32,3,3,${STRIDES_AND_DILATIONS})
}
# Batch size 1, 32 channels, stride 2, dilation 2.
function depthwise_conv_2d_static_small_stride_2_dilation_2() {
  STRIDES_AND_DILATIONS='[2,2],[2,2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,16,16,32,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,26,38,32,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,56,74,32,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,100,113,32,3,3,${STRIDES_AND_DILATIONS})
}

# Batch size 1, 128 channels, stride 1, dilation 1.
function depthwise_conv_2d_static_medium_stride_1_dilation_1() {
  STRIDES_AND_DILATIONS='[1,1],[1,1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,16,16,128,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,26,38,128,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,56,74,128,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,100,113,128,3,3,${STRIDES_AND_DILATIONS})
}
# Batch size 1, 128 channels, stride 2, dilation 1.
function depthwise_conv_2d_static_medium_stride_2_dilation_1() {
  STRIDES_AND_DILATIONS='[2,2],[1,1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,16,16,128,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,26,38,128,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,56,74,128,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,100,113,128,3,3,${STRIDES_AND_DILATIONS})
}
# Batch size 1, 128 channels, stride 1, dilation 2.
function depthwise_conv_2d_static_medium_stride_1_dilation_2() {
  STRIDES_AND_DILATIONS='[1,1],[2,2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,16,16,128,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,26,38,128,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,56,74,128,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,100,113,128,3,3,${STRIDES_AND_DILATIONS})
}
# Batch size 1, 128 channels, stride 1, dilation 2.
function depthwise_conv_2d_static_medium_stride_2_dilation_2() {
  STRIDES_AND_DILATIONS='[2,2],[2,2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,16,16,128,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,26,38,128,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,56,74,128,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,100,113,128,3,3,${STRIDES_AND_DILATIONS})
}

# Batch size 1, 1024 channels, stride 1, dilation 1.
function depthwise_conv_2d_static_large_stride_1_dilation_1() {
  STRIDES_AND_DILATIONS='[1,1],[1,1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,16,16,1024,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,26,38,1024,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,56,74,1024,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,100,113,1024,3,3,${STRIDES_AND_DILATIONS})
}
# Batch size 1, 1024 channels, stride 2, dilation 1.
function depthwise_conv_2d_static_large_stride_2_dilation_1() {
  STRIDES_AND_DILATIONS='[2,2],[1,1]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,16,16,1024,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,26,38,1024,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,56,74,1024,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,100,113,1024,3,3,${STRIDES_AND_DILATIONS})
}
# Batch size 1, 1024 channels, stride 1, dilation 2.
function depthwise_conv_2d_static_large_stride_1_dilation_2() {
  STRIDES_AND_DILATIONS='[1,1],[2,2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,16,16,1024,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,26,38,1024,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,56,74,1024,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,100,113,1024,3,3,${STRIDES_AND_DILATIONS})
}
# Batch size 1, 1024 channels, stride 1, dilation 2.
function depthwise_conv_2d_static_large_stride_2_dilation_2() {
  STRIDES_AND_DILATIONS='[2,2],[2,2]'
  COMMAND="cset proc -s sandbox -e python -- -m python.examples.depthwise_conv.depthwise_conv_2d_bench ${DUMP_DATA_FLAG} --dynamic_at_compile_time_list []"
  (${COMMAND} --problem_sizes_list 1,16,16,1024,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,26,38,1024,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,56,74,1024,3,3,${STRIDES_AND_DILATIONS})
  (${COMMAND} --problem_sizes_list 1,100,113,1024,3,3,${STRIDES_AND_DILATIONS})
}

###############################################################################
# Entry points.
###############################################################################
function run_all() {
  BENCH_ROOT_DIR=$(dirname $0)/benchmarks
  BENCH_DIR=${BENCH_ROOT_DIR}/results_$(ls -l ${BENCH_ROOT_DIR} | wc -l)
  mkdir -p ${BENCH_DIR}

  for benchmark in \
    copy_2d_static_small \
        copy_2d_static_small_2 \
        copy_2d_static_small_3 \
    transpose_2d_static_small \
    reduction_1d_static_small \
        row_reduction_2d_static_small \
        column_reduction_2d_static_small \
    matmul_static_small \
        matmul_static_small_reduction_dimension \
        matmul_static_medium \
        matmul_static_large \
    conv_1d_static_small_stride_1_dilation_1 \
        conv_1d_static_small_stride_2_dilation_1 \
        conv_1d_static_small_stride_1_dilation_2 \
        conv_1d_static_small_stride_2_dilation_2 \
    conv_1d_static_medium_stride_1_dilation_1 \
        conv_1d_static_medium_stride_2_dilation_1 \
        conv_1d_static_medium_stride_1_dilation_2 \
        conv_1d_static_medium_stride_2_dilation_2 \
    conv_1d_static_large_stride_1_dilation_1 \
        conv_1d_static_large_stride_2_dilation_1 \
        conv_1d_static_large_stride_1_dilation_2 \
        conv_1d_static_large_stride_2_dilation_2 \
    conv_2d_static_small_stride_1_dilation_1 \
        conv_2d_static_small_stride_2_dilation_1 \
        conv_2d_static_small_stride_1_dilation_2 \
        conv_2d_static_small_stride_2_dilation_2 \
    conv_2d_static_medium_stride_1_dilation_1 \
      conv_2d_static_medium_stride_2_dilation_1 \
      conv_2d_static_medium_stride_1_dilation_2 \
      conv_2d_static_medium_stride_2_dilation_2 \
    conv_2d_static_large_stride_1_dilation_1 \
      conv_2d_static_large_stride_2_dilation_1 \
      conv_2d_static_large_stride_1_dilation_2 \
      conv_2d_static_large_stride_2_dilation_2 \
    depthwise_conv_1d_static_small_stride_1_dilation_1 \
        depthwise_conv_1d_static_small_stride_2_dilation_1 \
        depthwise_conv_1d_static_small_stride_1_dilation_2 \
        depthwise_conv_1d_static_small_stride_2_dilation_2 \
    depthwise_conv_1d_static_medium_stride_1_dilation_1 \
        depthwise_conv_1d_static_medium_stride_2_dilation_1 \
        depthwise_conv_1d_static_medium_stride_1_dilation_2 \
        depthwise_conv_1d_static_medium_stride_2_dilation_2 \
    depthwise_conv_1d_static_large_stride_1_dilation_1 \
        depthwise_conv_1d_static_large_stride_2_dilation_1 \
        depthwise_conv_1d_static_large_stride_1_dilation_2 \
        depthwise_conv_1d_static_large_stride_2_dilation_2 \
    depthwise_conv_2d_static_small_stride_1_dilation_1 \
        depthwise_conv_2d_static_small_stride_2_dilation_1 \
        depthwise_conv_2d_static_small_stride_1_dilation_2 \
        depthwise_conv_2d_static_small_stride_2_dilation_2 \
    depthwise_conv_2d_static_medium_stride_1_dilation_1 \
        depthwise_conv_2d_static_medium_stride_2_dilation_1 \
        depthwise_conv_2d_static_medium_stride_1_dilation_2 \
        depthwise_conv_2d_static_medium_stride_2_dilation_2 \
    depthwise_conv_2d_static_large_stride_1_dilation_1 \
        depthwise_conv_2d_static_large_stride_2_dilation_1 \
        depthwise_conv_2d_static_large_stride_1_dilation_2 \
        depthwise_conv_2d_static_large_stride_2_dilation_2
  do
      unset SANDBOX_INLINING
      PLOT_NAME=""
      PEAK_COMPUTE="192"
      PEAK_BANDWIDTH="281"
      echo ${benchmark} ${BENCH_DIR}/${benchmark}.data ${BENCH_DIR}/${benchmark}.pdf
      prepare_data_collection ${BENCH_DIR}/${benchmark}.data
      ${benchmark}
      python ./tools/plot_benchmark.py --input ${BENCH_DIR}/${benchmark}.data \
        --output ${BENCH_DIR}/${benchmark}.pdf --name "${benchmark}" \
        --peak_compute ${PEAK_COMPUTE} --peak_bandwidth_hi ${PEAK_BANDWIDTH} \
        --peak_bandwidth_lo ${PEAK_BANDWIDTH}
  done
}

function run_and_plot_one() {
  BENCH_ROOT_DIR=$(dirname $0)/benchmarks
  BENCH_DIR=${BENCH_ROOT_DIR}/results_$(ls -l ${BENCH_ROOT_DIR} | wc -l)
  mkdir -p ${BENCH_DIR}

  unset SANDBOX_INLINING
  PLOT_NAME=""
  PEAK_COMPUTE="192"
  # L1 values
  PEAK_BANDWIDTH_HI="281"
  PEAK_BANDWIDTH_LO="281"
  # L2 values
  # PEAK_BANDWIDTH_HI="81"
  # PEAK_BANDWIDTH_LO="48"
  # L3 values
  # PEAK_BANDWIDTH_HI="26"
  # PEAK_BANDWIDTH_LO="16"
  benchmark=$1
  echo ${benchmark} ${BENCH_DIR}/${benchmark}.data ${BENCH_DIR}/${benchmark}.pdf
  prepare_data_collection ${BENCH_DIR}/${benchmark}.data
  ${benchmark}
  python ./tools/plot_benchmark.py --input ${BENCH_DIR}/${benchmark}.data \
    --output ${BENCH_DIR}/${benchmark}.pdf --name "${benchmark}" \
    --peak_compute ${PEAK_COMPUTE} --peak_bandwidth_hi ${PEAK_BANDWIDTH_HI} \
    --peak_bandwidth_lo ${PEAK_BANDWIDTH_LO}
}