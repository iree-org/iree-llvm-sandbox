# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Schedules for JAX.

load("@rules_python//python:defs.bzl", "py_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@llvm-project//mlir:build_defs.bzl", "cc_headers_only")
load("@rules_license//rules:license.bzl", "license")

license(
    name = "license",
    package_name = "Jasc",
    license_text = "LICENSE",
    package_url = "https://github.com/iree-org/iree-llvm-sandbox/tree/main/jasc",
)

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "jasc",
    srcs = [
        "__init__.py",
        "jasc.py",
    ],
    deps = [
        ":call_kernel",
        ":primitives",
        "//dialect:python",
        "//transform_ops",
        "@jax1//jax",
        "@jax1//jaxlib/mlir:bufferization_dialect",
        "@jax1//jaxlib/mlir:core",
        "@jax1//jaxlib/mlir:ir",
        "@jax1//jaxlib/mlir:pdl_dialect",
        "@jax1//jaxlib/mlir:transform_dialect",
    ],
)

py_library(
    name = "tuner",
    srcs = ["tuner.py"],
    deps = [
        ":jasc",
        "//transform_ops",
        "@jax1//jax",
        "@jax1//jaxlib/mlir:ir",
        "@jax1//jaxlib/mlir:transform_dialect",
    ],
)

py_library(
    name = "primitives",
    srcs = ["primitives.py"],
    deps = [
        ":call_kernel",
        "//dialect:python",
        "@jax1//jax",
        "@jax1//jax:extend",
        "@jax1//jaxlib/mlir:ir",
        "@jax1//jaxlib/mlir:pdl_dialect",
        "@jax1//jaxlib/mlir:stablehlo_dialect",
        "@jax1//jaxlib/mlir:transform_dialect",
    ],
)

cc_binary(
    name = "libmlir_c_runner_utils.so",
    linkopts = [
        "-Wl,-soname=libmlir_c_runner_utils.so",
        "-Wl,-rpath='$$ORIGIN'",
    ],
    linkshared = 1,
    deps = ["@llvm-project//mlir:mlir_c_runner_utils"],
)

pybind_extension(
    name = "call_kernel",
    srcs = ["call_kernel.cc"],
    deps = [
        ":libmlir_c_runner_utils.so",
        ":mlir_lowering_shared_library",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@jax1//jaxlib/mlir/_mlir_libs:jaxlib_mlir_capi_shared_library",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MLIRBindingsPythonHeaders",
        "@pybind11_abseil//pybind11_abseil:import_status_module",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@status_macros",
    ],
)

#
# `mlir_lowering` library.
#
# 1. Dependencies only. This allows to get the headers of all dependencies.
cc_library(
    name = "mlir_lowering_shared_library_deps",
    visibility = ["//visibility:private"],
    deps = [
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:AllToLLVMIRTranslations",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToGPURuntimeTransforms",
        "@llvm-project//mlir:GPUToNVVMTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:IndexToLLVM",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:SerializeToCubin_stub",
        "@llvm-project//mlir:SparseTensorTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformDialectTransforms",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorToLLVM",
        "@llvm-project//mlir:VectorToSCF",
        "@xla//xla/mlir_hlo:mhlo_passes",
    ],
)

cc_headers_only(
    name = "mlir_lowering_shared_library_deps_headers",
    src = "mlir_lowering_shared_library_deps",
    visibility = ["//visibility:private"],
)

# 2. The main library. This shouldn't be used directly in `py_extension`s.
cc_library(
    name = "mlir_lowering",
    srcs = [
        "gpu_lowering_passes.cc",
        "mlir_lowering.cc",
    ],
    hdrs = [
        "gpu_lowering_passes.h",
        "mlir_lowering.h",
    ],
    data = ["gpu_post_bufferize.mlir"],
    visibility = [
        # `jaxlib_mlir_capi_shared_library` needs to depend on `mlir_lowering`
        # because (1) it depends on other targets that need symbols from this
        # target and (2) that target cannot depend on
        # `mlir_lowering_shared_library` because the reverse dependency must
        # exist (since, otherwise, `mlir_lowering_shared_library` would
        # duplicate symbols from `jaxlib_mlir_capi_shared_library`).
        "@jax1//jaxlib/mlir/_mlir_libs:__pkg__",
    ],
    deps = [
        ":mlir_lowering_shared_library_deps_headers",
        # Only depend on the headers here to avoid duplicate symbols.
        "//dialect:dialect_headers",
        "//transform_ops:jasc_transform_ops_headers",
    ],
    # This is important since it makes sure that the symbols of the library are
    # exported by the `.so` target below even though they aren't used directly.
    alwayslink = True,
)

cc_headers_only(
    name = "mlir_lowering_headers",
    src = "mlir_lowering",
    visibility = ["//visibility:private"],
)

# 3. Shared object file. This forces to create a shared library, which dependent
# targets can link against, instead of using the default static linking. This
# ensures that the symbols in that library exist only once instead of once for
# each time it is linked statically.
# This pattern is copied from JAX. A platform independent version exists there.
cc_binary(
    name = "libmlirlowering.so",
    linkopts = [
        "-Wl,-soname=libmlirlowering.so",
        "-Wl,-rpath='$$ORIGIN'",
    ],
    linkshared = 1,
    visibility = ["//visibility:private"],
    deps = [
        ":mlir_lowering",
        "@jax1//jaxlib/mlir/_mlir_libs:jaxlib_mlir_capi_shared_library",
    ],
)

# 4. A `cc_library` wrapper of the shared library. This is the main target.
cc_library(
    name = "mlir_lowering_shared_library",
    srcs = [
        "mlir_lowering.h",
        ":libmlirlowering.so",
    ],
    deps = [
        ":mlir_lowering_headers",
        ":mlir_lowering_shared_library_deps_headers",
        "@jax1//jaxlib/mlir/_mlir_libs:jaxlib_mlir_capi_shared_library",
    ],
)

cc_binary(
    name = "jasc-opt",
    srcs = ["jasc_opt.cc"],
    deps = [
        ":mlir_lowering",
        "//dialect",
        "//transform_ops:jasc_transform_ops",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:AllExtensions",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:AllToLLVMIRTranslations",
        "@llvm-project//mlir:MlirOptLib",
        "@xla//xla/mlir_hlo:mhlo_passes",
    ],
)
