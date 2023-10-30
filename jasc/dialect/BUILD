# MLIR Dialect to support Jasc transformations.

load("@rules_python//python:defs.bzl", "py_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")
load("@llvm-project//mlir:build_defs.bzl", "cc_headers_only")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    # default_applicable_licenses = ["//third_party/mlir_edge:license"],
    default_visibility = ["//visibility:public"],
)

td_library(
    name = "td_files",
    srcs = [
        "dialect.td",
        "ops.td",
    ],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:TransformDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ops.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

cc_library(
    name = "jasc_dialect_shared_library_deps",
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:TransformDialect",
    ],
)

cc_headers_only(
    name = "jasc_dialect_shared_library_deps_headers",
    src = "jasc_dialect_shared_library_deps",
)

cc_headers_only(
    name = "jasc_dialect_headers",
    src = "dialect",
)

cc_library(
    name = "dialect",
    srcs = [
        "dialect.cc",
        "ops.cc",
    ],
    hdrs = [
        "dialect.h",
        "ops.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":jasc_dialect_shared_library_deps_headers",
    ],
    alwayslink = True,
)

cc_library(
    name = "capi",
    srcs = [
        "capi.cc",
    ],
    hdrs = [
        "capi.h",
    ],
    deps = [
        ":dialect",
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
    alwayslink = True,
)

cc_library(
    name = "capi_headers",
    hdrs = [
        "capi.h",
    ],
    deps = [
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
)

gentbl_filegroup(
    name = "ops_py_gen",
    tbl_outs = [
        (
            [
                "-gen-python-op-bindings",
                "-bind-dialect=jasc",
            ],
            "_ops_gen.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ops_py.td",
    deps = [
        ":td_files",
    ],
)

cc_library(
    name = "jasc_dialect_shared_library",
    srcs = [
        ":libjascdialect.so",
        "dialect.h",
        "ops.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":jasc_dialect_shared_library_deps_headers",
    ],
)

cc_binary(
    name = "libjascdialect.so",
    linkopts = [
        "-Wl,-soname=libjascdialect.so",
        "-Wl,-rpath='$$ORIGIN'",
    ],
    linkshared = 1,
    deps = [":dialect"],
)

pybind_extension(
    name = "bindings",
    srcs = ["bindings.cc"],
    deps = [
        "@jax1//jaxlib/mlir/_mlir_libs:jaxlib_mlir_capi_shared_library",
        ":jasc_dialect_headers",
        "@llvm-project//mlir:CAPIIRHeaders",
        "@llvm-project//mlir:MLIRBindingsPythonHeaders",
        "//:mlir_lowering_shared_library",
        "//transform_ops:jasc_transform_ops_shared_library",
    ],
)

py_library(
    name = "python",
    srcs = [
        "jasc.py",
        # "_ods_common.py",
        ":ops_py_gen",
    ],
    deps = [
        ":bindings",
        # "@jax//jaxlib/mlir:core",
        # "@jax//jaxlib/mlir:ir",
        # "@jax//jaxlib/mlir:pdl_dialect",
    ],
)
