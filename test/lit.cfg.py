import os
import sys

import lit.formats
import lit.util

import lit.llvm

# Configuration file for the 'lit' test runner.
lit.llvm.initialize(lit_config, config)

# name: The name of this test suite.
config.name = "mlir-proto-opt tests"

config.test_format = lit.formats.ShTest(execute_external=True)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The path where tests shall execute.
build_dir = os.environ["IREE_LLVM_SANDBOX_BUILD_DIR"]
config.test_exec_root = os.path.join(build_dir, 'test')
lib_dir = os.path.join(build_dir, 'lib')

#config.use_default_substitutions()
config.excludes = [
  "lit.cfg.py",
  "lit.site.cfg.py",

  # Currently disabled tests.
  "tiling.mlir",
  "constant.mlir",
  "test_matmul_f32_cuda.mlir",
  "matmul-f32-mt-cpu.mlir",
  "pack-2d-to-4d-by-8x4-blocks.mlir",
  "test_matmul_f16_cuda_mma.mlir",
  "vector-distribution.mlir",
  "matmul-f32-base.mlir",
  # FIXME: Enable after landing D121369.
  "double-tiling.mlir",
  "tile-interchange.mlir"
]

config.substitutions.extend([
    ("%PYTHON", sys.executable),
    ("%mlir_runner_utils_dir", lib_dir),
    ("%shlibext", ".so"),
])

# Add the build/bin directory to the path.
sys.path.append(os.path.join(build_dir, "bin"))
config.environment["PYTHONPATH"] = ":".join(sys.path)
config.environment["PATH"] = ":".join(sys.path)
project_root = os.path.dirname(os.path.dirname(__file__))
