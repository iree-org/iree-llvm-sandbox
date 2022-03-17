# How to enable and use alp

This is a very simple set of instructions to enable and work with alp from iree-llvm-sandbox. We use the following environment variables defaults in these instructions:

* `IREE_LLVM_SANDBOX_SOURCE_DIR`: path to the source of the iree-llvm-sandbox
* `IREE_LLVM_SANDBOX_BUILD_DIR`: path to the source of the iree-llvm-sandbox
* `LLVM_SOURCE_DIR`: path to the source of the llvm-project folder

We also need to set the correct `$PYTHONPATH` to enable the python infrastructure:

```
export PYTHONPATH=$IREE_LLVM_SANDBOX_SOURCE_DIR/build/tools/sandbox/python_packages:$IREE_LLVM_SANDBOX_SOURCE_DIR/python/examples/:$LLVM_SOURCE_DIR/mlir/python:$IREE_LLVM_SANDBOX_SOURCE_DIR/experimental/alp/python
 ```

## Download LLVM

You should clone LLVM and point it to the commit indicated in: `${IREE_LLVM_SOURCE_DIR}/pinned-llvm-version`

```
git clone https://github.com/llvm/llvm-project.git
cd llvm-project 
git checkout `cat ${IREE_LLVM_SANDBOX_SOURCE_DIR}/pinned-llvm-version`
```

## Download LLVM [Internal development]

For internal development you should clone directly from the codehub mirror:

```
git clone ssh://git@codehub-dg-y.huawei.com:2222/boole-compiler/uk-team/llvm-project.git
git checkout main
```

## Build all together

This needs to be run from $IREE_LLVM_SANDBOX_SOURCE_DIR. I am pointing out instructions for AArch64 + ALP:

```
python3 ./configure.py --target=AArch64 --llvm-path=$LLVM_SOURCE_DIR --alp
```

 Please note that the supported `cmake` version is >= 3.21.0

After this command, if you only want to rebuild, you can simply do:

```
cmake --build $IREE_LLVM_SANDBOX_SOURCE_DIR/build --target tools/sandbox/all mlir-opt mlir-translate mlir_runner_utils mlir_c_runner_utils llvm-mca llvm-objdump llc opt
 ```

## Use the tool

Given a generic MLIR program, `prog.mlir`, we can compile it in the following way:

```
python3 -m alp.backend.mlirc --input-file=prog.mlir ... # transformation flags
```

This will create an assembly file `prog.s`. In order to run it, we have two options:
a) Link the assembly to a C++ program (see Transition Path below), link and run
b) Write a benchmark program in MLIR and execute it through the python framework.

In this section, we will show-case option b) using GEMM as an example. In the following we assume that the current folder is `$IREE_LLVM_SANDBOX_SOURCE_DIR/experimental/alp`

### Generate the target program

Our transition python module is supposed to generate MLIR program for known library functions. To generate GEMM, you can run:

```
python3 -m alp.transition.blas.gemm --M 2048 --N 2048 --K 2048 --trA
```

This will generate a `gemm.mlir` program in the current folder which is supposed to execute a matrix multiply operation `C += A*B` where `A` is pre-transposed. You can also generate a dynamic sized GEMM by not specifying any of the sizes. For instance:

```
python3 -m alp.transition.blas.gemm  --trA
```

Generates a fully dynamic GEMM implementation where the sizes are read dynamically from the inputs.

### Compile the program

We can compile `gemm.mlir` in the following way:

```
python3 -m alp.backend.mlirc --input-file=gemm.mlir --tile-sizes 2048 512 128 --register-tile-sizes 8 8 1 --reorder-tile-sizes 0 2 1 --reorder-register-tile-sizes 0 1 2 --unroll-vector-transfers --split-vector-transfers-to none --hoist-packing 4 3 0 --modulo-scheduling --ms-unroll=2 --transpose-packing 0 0 0 --verbosity-level=4
```

A file `gemm.s` should be created in your current folder.

### Benchmark the program

Our infrastructure provides the possibility to generate a benchmark MLIR file, compile it, link it with the target assembly file and run it. This is what you have to do:

```
python3 -m alp.benchmark.blas.gemm --asm-program=gemm.s --M=2048 --N=2048 --K=2048 --trA
```

Please note that in this case we need to provide information to the benchmark about what we want to run. If you want to re-run the benchmark you can either issue the same command again, or you can simply run the executable `gemm.bench.out` that has been created in your current folder. You may also want to just generate the benchmark program, and in this case you should simply run:

```
python3 -m alp.benchmark.blas.gemm --M=2048 --N=2048 --K=2048 --trA
```

Also, you can have a look at the `gemm.bench.mlir` file that has been generated within your current folder.

### Test the program

You can finally test that the transformed program is correct. The command is very similar to the ones using for benchmarking:

```
python3 -m alp.test.blas.gemm --asm-program=gemm.s --M=2048 --N=2048 --K=2048 --trA
```

Please note that we are using a naive algorithm to compute the matrix multiply, and this might take some time to finish.

### Smoke test

```
cd $IREE_LLVM_SANDBOX_SOURCE_DIR/experimental/alp
make check
```

## Use the tuner

### Download OpenTuner

OpenTuner should come as a prebuild package installable directly from `pip3`:

```
pip3 install  --user  opentuner
```

### Tune a gemm program

The tuner is the real backend compiler, since it issues the transformations to apply to the program via `mlirc`. To run the tuner needs:

* The MLIR program to compile
* The MLIR benchmark to execute the program

```
python3 -m alp.backend.tuner --input-file gemm.mlir --benchmark gemm.bench.mlir
```
