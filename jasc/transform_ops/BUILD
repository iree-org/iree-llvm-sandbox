# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# JASC extension for the MLIR transform dialect.

load("@rules_python//python:defs.bzl", "py_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")
load("@llvm-project//mlir:build_defs.bzl", "cc_headers_only")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:public"],
)

td_library(
    name = "td_files",
    srcs = glob(["*.td"]),
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:TransformDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "jasc_transform_ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "jasc_transform_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "jasc_transform_ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "jasc_transform_ops.td",
    deps = [":td_files"],
)

gentbl_filegroup(
    name = "jasc_transform_ops_py_gen",
    tbl_outs = [(
        [
            "-gen-python-op-bindings",
            "-bind-dialect=transform",
            "-dialect-extension=jasc_transform",
        ],
        "_jasc_transform_ops_gen.py",
    )],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "jasc_transform_ops.td",
    deps = [":td_files"],
)

pybind_extension(
    name = "bindings",
    srcs = ["bindings.cpp"],
    deps = [
        "@jax1//jaxlib/mlir/_mlir_libs:jaxlib_mlir_capi_shared_library",
        ":jasc_transform_ops_shared_library",
        "@llvm-project//mlir:CAPIIRHeaders",
        "@llvm-project//mlir:MLIRBindingsPythonHeaders",
    ],
)

py_library(
    name = "transform_ops",
    srcs = [
        "_ods_common.py",
        "_transform_ops_gen.py",
        "jasc_transform_ops.py",
        ":jasc_transform_ops_py_gen",
    ],
    deps = [
        ":bindings",
    ],
)

cc_library(
    name = "jasc_transform_ops_shared_library_deps",
    deps = [
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformDialect",
    ],
)

cc_headers_only(
    name = "jasc_transform_ops_shared_library_deps_headers",
    src = "jasc_transform_ops_shared_library_deps",
)

cc_library(
    name = "jasc_transform_ops_shared_library",
    srcs = [
        ":libjasctransformops.so",
    ],
    hdrs = glob(["*.h"]),
    deps = [
        "jasc_transform_ops_shared_library_deps_headers",
    ],
)

cc_headers_only(
    name = "jasc_transform_ops_shared_library_headers",
    src = "jasc_transform_ops_shared_library",
)

cc_binary(
    name = "libjasctransformops.so",
    linkopts = [
        "-Wl,-soname=libjasctransformops.so",
        "-Wl,-rpath='$$ORIGIN'",
    ],
    linkshared = 1,
    deps = [":jasc_transform_ops"],
)

cc_headers_only(
    name = "jasc_transform_ops_headers",
    src = "jasc_transform_ops",
)

cc_library(
    name = "jasc_transform_ops",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    includes = ["."],
    deps = [
        ":jasc_transform_ops_inc_gen",
        ":jasc_transform_ops_shared_library_deps_headers"
    ],
    alwayslink = True,
)
