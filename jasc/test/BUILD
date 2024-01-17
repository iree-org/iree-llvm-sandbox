# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# JASC filecheck tests

load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@llvm-project//llvm:lit_test.bzl", "lit_test", "package_path")
load("@pip_deps//:requirements.bzl", "requirement")
load("@bazel_skylib//rules:expand_template.bzl", "expand_template")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

# The image needs a library target that is not testonly.
# TODO(zinenko): consolidate as filegroups.
py_library(
    name = "cpu_integration_lib",
    srcs = ["cpu_integration.py"],
    tags = ["manual"],
    deps = [
        "//:jasc",
        requirement("chex"),
        requirement("pytest"),
        "@jax1//jax",
    ],
)

py_library(
    name = "gpu_integration_lib",
    srcs = ["gpu_integration.py"],
    tags = ["manual"],
    deps = [
        ":cpu_integration_lib",
        requirement("chex"),
        requirement("pytest"),
    ],
)

# The image needs a binary target that is not testonly.
# TODO(zinenko): consolidate as filegroups.
py_binary(
    name = "gpu_integration_binary",
    srcs = ["gpu_integration.py"],
    main = "gpu_integration.py",
    tags = ["manual"],
    deps = [
        ":gpu_integration_lib",
    ],
)

py_test(
    name = "gpu_integration",
    srcs = [
        "gpu_integration.py",
    ],
    tags = ["requires-gpu-nvidia"],
    deps = [
        ":gpu_integration_lib",
    ],
)

py_test(
    name = "cpu_integration",
    srcs = ["cpu_integration.py"],
    deps = [
        ":cpu_integration_lib",
    ],
)

py_binary(
    name = "abstractions",
    srcs = ["abstractions.py"],
    deps = [
        "//:jasc",
        "//transform_ops",
        requirement("absl-py"),
        requirement("ml_dtypes"),
        requirement("opt_einsum"),
        "@jax1//jaxlib/mlir:ir",
        "@jax1//jaxlib/mlir:scf_dialect",
    ],
)

py_binary(
    name = "diagnostics",
    srcs = ["diagnostics.py"],
    deps = [
        "//:jasc",
        requirement("absl-py"),
        requirement("ml_dtypes"),
        requirement("opt_einsum"),
        "@jax1//jax",
        "@jax1//jaxlib/mlir:ir",
    ],
)

py_binary(
    name = "normalization",
    srcs = ["normalization.py"],
    deps = [
        "//:jasc",
        "//transform_ops",
        requirement("absl-py"),
        requirement("ml_dtypes"),
        requirement("opt_einsum"),
        "@jax1//jaxlib/mlir:ir",
        "@jax1//jaxlib/mlir:scf_dialect",
        "@jax1//jaxlib/mlir:transform_dialect",
    ],
)

py_binary(
    name = "tag",
    srcs = ["tag.py"],
    deps = [
        "//:jasc",
        requirement("absl-py"),
        requirement("chex"),
        "@jax1//jax",
    ],
)

py_test(
    name = "batch_matmul_gpu",
    srcs = ["batch_matmul_gpu.py"],
    tags = ["requires-gpu-nvidia"],
    deps = [
        "//:jasc",
        requirement("chex"),
        requirement("pytest"),
        "@jax1//jax",
        "@jax1//jaxlib/mlir:ir",
        "@jax1//jaxlib/mlir:transform_dialect",
    ],
)

py_test(
    name = "matmul_gpu",
    srcs = ["matmul_gpu.py"],
    tags = ["requires-gpu-nvidia"],
    deps = [
        "//:jasc",
        requirement("chex"),
        requirement("pytest"),
        "@jax1//jax",
        "@jax1//jaxlib/mlir:ir",
        "@jax1//jaxlib/mlir:transform_dialect",
    ],
)

py_test(
    name = "matmul_cpu",
    srcs = ["matmul_cpu.py"],
    deps = [
        "//:jasc",
        requirement("absl-py"),
        requirement("chex"),
        requirement("pytest"),
        "@jax1//jax",
        "@jax1//jaxlib/mlir:transform_dialect",
    ],
)

py_test(
    name = "autotuning",
    srcs = ["autotuning.py"],
    deps = [
        "//:jasc",
        "//:tuner",
        requirement("absl-py"),
        requirement("chex"),
        requirement("pytest"),
        "@jax1//jax",
        "@jax1//jaxlib/mlir:ir",
        "@jax1//jaxlib/mlir:linalg_dialect",
    ],
)

py_test(
    name = "jit",
    srcs = ["jit.py"],
    deps = [
        "//:jasc",
        requirement("chex"),
        requirement("pytest"),
        "@jax1//jax",
    ],
)

py_binary(
    name = "bindings",
    srcs = ["bindings.py"],
    deps = [
        "//:jasc",
        "//transform_ops",
        requirement("absl-py"),
        "@jax1//jaxlib/mlir:ir",
        "@jax1//jaxlib/mlir:pass_manager",
        "@jax1//jaxlib/mlir:transform_dialect",
    ],
)

[sh_test(
    name = target + "_filecheck_test",
    srcs = ["filecheck_test.sh"],
    args = [target],
    data = [
        ":" + target,
        "@llvm-project//llvm:FileCheck",
    ],
) for target in [
    "abstractions",
    "bindings",
    "diagnostics",
    "normalization",
    "tag",
]]

LLVM_LIT_PATH_FUNCTION = " " + \
                         "# Allow generated file to be relocatable.\n" + \
                         "from pathlib import Path\n" + \
                         "def path(p):\n" + \
                         "    if not p: return ''\n" + \
                         "    return str((Path(__file__).parent / p).resolve())\n"

LIT_SITE_CFG_IN_HEADER = "# Autogenerated, do not edit.\n\n" + LLVM_LIT_PATH_FUNCTION

expand_template(
    name = "lit_site_cfg_py",
    testonly = True,
    out = "lit.site.cfg.py",
    substitutions = {
        "@LIT_SITE_CFG_IN_HEADER@": LIT_SITE_CFG_IN_HEADER,
        "@LLVM_TOOLS_DIR@": package_path("@llvm-project//llvm:BUILD"),
        "@MLIR_TOOLS_DIR@": package_path("@llvm-project//mlir:BUILD"),
        "@SHLIBDIR@": package_path("@llvm-project//llvm:BUILD"),
        "@JASC_SOURCE_DIR@": package_path("@jasc//:BUILD"),
        "@JASC_TOOLS_DIR@": package_path("@jasc//:jasc-opt"),
    },
    template = "lit.site.cfg.in.py",
)

filegroup(
    name = "lit_data",
    testonly = True,
    data = [
        "lit.cfg.py",
        "lit.site.cfg.py",
        "//:jasc-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:count",
        "@llvm-project//llvm:not",
    ],
)

[
    lit_test(
        name = "%s.test" % src,
        srcs = [src],
        data = [
            ":lit_data",
        ],
    )
    for src in glob(
        include = ["*.mlir"],
    )
]
