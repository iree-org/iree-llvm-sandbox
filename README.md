# IREE LLVM Sandbox

DISCLAIMER: This is not an officially-supported Google project. It is a sandbox
for quick iteration and experimentation on projects related to the IREE project,
MLIR, and LLVM.

This repository contains experimental work by the IREE team closely related to
LLVM and MLIR, usually with the aim of upstreaming in some form. The main
project is at <https://github.com/google/iree>.

As an experimental project, build greenness, documentation, and polish are
likely to be minimal, as it instead prioritizes easy experimentation.

# License

Licensed under the Apache license with LLVM Exceptions. See [LICENSE](LICENSE)
for more information.

# Build instructions

This project builds as part of the LLVM External Projects facility (see
documentation for the `LLVM_EXTERNAL_PROJECTS` config setting). There are many
ways to set this up. We recommend using our `configure.py` script below.

It is left to the reader to adapt paths if deviating. We assume below that
projects are checked out to `$HOME/src`.

## Check out projects

TODO: Simplify instructions.

In your `$HOME/src` directory, check out each project:

Required:

* `git clone https://github.com/google/iree-llvm-sandbox`

We use the following environment variables defaults in these instructions:

* `IREE_LLVM_SANDBOX_SOURCE_DIR`: $HOME/src/iree-llvm-sandbox
* `IREE_LLVM_SANDBOX_BUILD_DIR`: ${IREE_LLVM_SANDBOX_SOURCE_DIR}/build

## Python prerequisites (if using Python)

Follow the instructions for
[MLIR Python Bindings](https://mlir.llvm.org/docs/Bindings/Python/):

```
which python
python -m venv ~/.venv/mlirdev
source ~/.venv/mlirdev/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configure and build

The sandbox can be optionally built with or without IREE integration (for
accessing IREE specific IR and evaluating on IREE compatible targets):

### Building with IREE

Checkout the [IREE](https://github.com/google/iree) GitHub repo next to this
directory and initialize submodules:

```
(cd .. && git clone https://github.com/google/iree --recurse-submodules=third_party/llvm-project)
```

And configure/build the project:

```
python configure.py --iree-path=../iree
```

Note that the `third_party/llvm-project` bundled with IREE will be used.

### Building without IREE

You must checkout [llvm-project](https://github.com/llvm/llvm-project) at a
compatible commit.

```
(cd .. && git clone https://github.com/llvm/llvm-project.git)
```

And configure/build the project. By default the `configure.py` script will look in `${IREE_LLVM_SANDBOX_SOURCE_DIR}/../llvm-project` (this can also
be overridden with `--llvm-path`):

```
python configure.py
```

## Using the Python API

```
source .env && export PYTHONPATH

# Sanity check (should not error).
python -c "import mlir.iree_sandbox"

# Run a matmul.
export MLIR_RUNNER_UTILS_LIB=${IREE_LLVM_SANDBOX_BUILD_DIR}/lib/libmlir_runner_utils.so; \
export MLIR_C_RUNNER_UTILS_LIB=${IREE_LLVM_SANDBOX_BUILD_DIR}/lib/libmlir_c_runner_utils.so; \
cd ${IREE_LLVM_SANDBOX_SOURCE_DIR}; \
python -m python.examples.matmul.test
```

## Using mlir-proto-opt

```
"${IREE_LLVM_SANDBOX_BUILD_DIR}"/bin/mlir-proto-opt \
  ../iree-llvm-sandbox/test/test_constant.mlir \
  -linalg-comprehensive-module-bufferize
```

TODOs:

1. hook up a lit test target.
2. re-add npcomp instructions once it is upgraded to use the same build setup.

## Running tests

The following commands either run the lit tests only or all tests:

```
# Run lit tests
lit -v test
# Run python and lit tests
python ./run_tests.py
```

The lit configuration file `test/lit.cfg.py` contains a list of excluded tests.

# Benchmark commands

Adaptation of recommended benchmark instructions found [here](https://llvm.org/docs/Benchmarking.html).
Run the following as root.

```
# Basic info
numactl --hardware

################################################################
# Prepare to run on a subset of CPUs only
################################################################
# Disable address space randomization.
echo 0 > /proc/sys/kernel/randomize_va_space

# Disable the sibling of CPU 4.
cat /sys/devices/system/cpu/cpu4/topology/thread_siblings_list

# E.g. on a 36 core system, this should return 4,40, use a shift of 36 for rest.
echo 0 > /sys/devices/system/cpu/cpu$((4 + 36))/online

# Disable the siblings of CPU 0-31, we'll use those for parallel runs.
for i in $(seq 0 31); do \
  echo 0 /sys/devices/system/cpu/cpu$(( ${i} + 36))/online; \
done

################################################################
# Perform cpuset manipulation.
################################################################
# For reference, cset shield does not seem to run as expected on at least 2 systems.
# cset shield -c 4 --user=${RUN_AS_USER} -k on --userset=${RUN_AS_USER}
# Instead, reproduce the following finer-grained instructions:
#   https://documentation.suse.com/sle-rt/15-SP2/html/SLE-RT-all/cha-shielding-cpuset.html

cset set -s system -c 32-35 -m 1

#for i in $(seq 0 32); do \
#  cset set -s sandbox_${i} -c ${i} -m 0 --cpu_exclusive
#done

cset set -s sandbox_parallel -c 0-31 -m 0 --cpu_exclusive

cset proc -m -f root -t system

################################################################
# Freq control (note, cloud VM instances do not allow).
################################################################

echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo
echo performance > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
for i in $(seq 0 31); do \
  echo performance > /sys/devices/system/cpu/cpu$(( ${i} ))/cpufreq/scaling_governor;\
done

################################################################
# Exec.
################################################################
IREE_LLVM_SANDBOX_BUILD_DIR=$(pwd)/build \
MLIR_RUNNER_UTILS_LIB=${IREE_LLVM_SANDBOX_BUILD_DIR}/lib/libmlir_runner_utils.so \
MLIR_C_RUNNER_UTILS_LIB=${IREE_LLVM_SANDBOX_BUILD_DIR}/lib/libmlir_c_runner_utils.so \
PYTHONPATH=${IREE_LLVM_SANDBOX_BUILD_DIR}/tools/sandbox/python_package cset proc -s sandbox \
-e ${PATH_TO_VENV}/.venv/mlirdev/bin/python -- -m python.examples.matmul.bench

IREE_LLVM_SANDBOX_BUILD_DIR=$(pwd)/build \
MLIR_RUNNER_UTILS_LIB=${IREE_LLVM_SANDBOX_BUILD_DIR}/lib/libmlir_runner_utils.so \
MLIR_C_RUNNER_UTILS_LIB=${IREE_LLVM_SANDBOX_BUILD_DIR}/lib/libmlir_c_runner_utils.so \
MLIR_C_RUNNER_UTILS_LIB=${IREE_LLVM_SANDBOX_BUILD_DIR}/lib/libmlir_c_runner_utils.so \
export MLIR_RUNNER_EXTRA_LIBS=${IREE_LLVM_SANDBOX_BUILD_DIR}/lib/libmlir_async_runtime_copy.so \
PYTHONPATH=${IREE_LLVM_SANDBOX_BUILD_DIR}/tools/sandbox/python_package cset proc -s sandbox_parallel \
-e ${PATH_TO_VENV}/.venv/mlirdev/bin/python -- -m python.examples.linalg_ext.in_par_bench
```
