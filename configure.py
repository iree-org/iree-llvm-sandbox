#!/usr/bin/env python
# Shortcut script to fetching deps and configuring the project for use.

import argparse
import os
import shutil
import subprocess
import sys

def parse_arguments():
  parser = argparse.ArgumentParser(description="Configure the project")
  parser.add_argument("--use-iree", 
                      help="Build with IREE support and IREE's LLVM (optional)",
                      default = False)
  parser.add_argument("--target",
                      help="Semicolumn-separated list of targets to build with LLVM",
                      default = "X86")
  parser.add_argument("--lld", 
                      help="Build with ENABLE_LLD=ON (optional)",
                      dest="enable_lld",
                      default = False)
  parser.add_argument("--no-ccache",
                      help="Disables ccache (if available)",
                      dest="enable_ccache",
                      action="store_false",
                      default=True)
  return parser.parse_args()


def main(args):
  # Sanity checks so we are all on the same page.
  assert "IREE_LLVM_SANDBOX_SOURCE_DIR" in os.environ, \
    "env var IREE_LLVM_SANDBOX_SOURCE_DIR must be set"
  assert "IREE_LLVM_SANDBOX_SOURCE_DIR" in os.environ and \
    os.getenv("IREE_LLVM_SANDBOX_SOURCE_DIR") == os.path.abspath(os.path.dirname(__file__)), \
    f"env var IREE_LLVM_SANDBOX_SOURCE_DIR must be set to {os.path.abspath(os.path.dirname(__file__))}"
  assert "IREE_LLVM_SANDBOX_BUILD_DIR" in os.environ, \
    "env var IREE_LLVM_SANDBOX_BUILD_DIR must be set"

  repo_root = os.path.abspath(os.getenv("IREE_LLVM_SANDBOX_SOURCE_DIR"))

  print(f"-- Using repo root {repo_root}")
  print(f"-- Python version {sys.version} ({sys.executable})")
  print(f"CCACHE = {args.enable_ccache}")

  llvm_projects = ["sandbox"]
  llvm_configure_args = [
      f"-DLLVM_EXTERNAL_SANDBOX_SOURCE_DIR={repo_root}",
  ]

  # TODO: Make configurable.
  llvm_builtin_projects = ["mlir", "clang", "clang-tools-extra"]

  # Detect and set the paths for IREE and LLVM.
  llvm_path, iree_path = None, None
  if args.use_iree:
    # Detect IREE (defaults LLVM path as well).
    # As per "Building with IREE" instructions:
    # You **must** checkout the [IREE](https://github.com/google/iree) GitHub repo next 
    # to this directory and initialize submodules:
    iree_path = os.path.join(repo_root, "..", "iree")
    print(f"-- Enabling IREE from {iree_path}")
    if not os.path.exists(os.path.join(iree_path, "CMakeLists.txt")):
      print(f"ERROR: Could not find iree at {iree_path}")
      return 1
    llvm_path = os.path.join(iree_path, "third_party", "llvm-project")
    iree_dialects_path = os.path.join(iree_path, "llvm-external-projects",
                                      "iree-dialects")
    if not os.path.exists(os.path.join(iree_dialects_path, "CMakeLists.txt")):
      print(f"ERROR: Cannot find iree-dialects project at {iree_dialects_path}")
      return 1
    # Must come before the sandbox project.
    llvm_projects.insert(0, "iree_dialects")
    #llvm_projects.append("iree_dialects")
    llvm_configure_args.append(
        f"-DLLVM_EXTERNAL_IREE_DIALECTS_SOURCE_DIR={iree_dialects_path}")
  else:
    # Detect LLVM.
    # As per "Building without IREE" instructions:
    # You **must** checkout [llvm-project](https://github.com/llvm/llvm-project) at a
    # compatible commit next to this directory.
    llvm_path = os.path.join(repo_root, "..", "llvm-project")
    print(f"-- Using default llvm-project path: {llvm_path}")
    if not os.path.exists(llvm_path):
      print(f"ERROR: Could not find llvm-project at {llvm_path}")
      return 1

  # Detect clang.
  clang_path = shutil.which("clang")
  clangpp_path = shutil.which("clang++")
  if clang_path and clangpp_path:
    llvm_configure_args.append(f"-DCMAKE_C_COMPILER={clang_path}")
    llvm_configure_args.append(f"-DCMAKE_CXX_COMPILER={clangpp_path}")
  else:
    print(
        "WARNING: Could not find clang. Building with default system compiler")

  # Detect lld.
  if args.enable_lld:
    lld_path = shutil.which("ld.lld")
    if lld_path:
      print(f"-- Using lld: {lld_path}")
      llvm_configure_args.append("-DLLVM_ENABLE_LLD=ON")
    else:
      print("WARNING: LLD (ld.lld) not found on path. Configure may fail.")

  # Detect ccache.
  if args.enable_ccache:
    ccache_path = shutil.which("ccache")
    if ccache_path:
      print(f"-- Using ccache: {ccache_path}")
      llvm_configure_args.append("-DLLVM_CCACHE_BUILD=ON")
    else:
      print("WARNING: Project developers use ccache which is not installed")

  # CMake configure.
  build_dir = os.path.abspath(os.getenv("IREE_LLVM_SANDBOX_BUILD_DIR"))
  os.makedirs(build_dir, exist_ok=True)
  cmake_args = [
      "cmake",
      "-GNinja",
      f"-B{build_dir}",
      f"-S{os.path.join(llvm_path, 'llvm')}",
      f"-DLLVM_ENABLE_PROJECTS={';'.join(llvm_builtin_projects)}",
      f"-DLLVM_TARGETS_TO_BUILD={args.target}",
      "-DMLIR_INCLUDE_INTEGRATION_TESTS=ON",
      "-DLLVM_ENABLE_ASSERTIONS=ON",
      "-DLLVM_INCLUDE_UTILS=ON",
      "-DLLVM_INSTALL_UTILS=ON",
      "-DLLVM_BUILD_EXAMPLES=ON",
      "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
      f"-DPython3_EXECUTABLE={sys.executable}",
      "-DCMAKE_BUILD_TYPE=Release",
      f"-DLLVM_EXTERNAL_PROJECTS={';'.join(llvm_projects)}",
  ] + llvm_configure_args
  print(f"-- Running cmake:\n  {' '.join(cmake_args)}")
  subprocess.check_call(cmake_args, cwd=build_dir)

  # Write out .env.
  with open(f"{os.path.join(repo_root, '.env')}", "wt") as f:
    f.write(f"PYTHONPATH={os.path.join(build_dir, 'tools', 'sandbox', 'python_package')}")

  # Do initial build.
  # Also build all the rest, including llvm-mca and libmlir_runner_utils so that 
  # everything is functional.
  # TODO: Remove as this may be too painful for CI.
  cmake_args = ["cmake", "--build", build_dir, "--target", "all"]
  print(f"-- Performing initial build: {' '.join(cmake_args)}")
  subprocess.check_call(cmake_args, cwd=build_dir)

  return 0


if __name__ == "__main__":
  sys.exit(main(parse_arguments()))
