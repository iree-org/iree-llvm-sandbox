#!/usr/bin/env python
# Shortcut script to fetching deps and configuring the project for use.

import argparse
import os
import shutil
import subprocess
import sys


def parse_arguments():
  parser = argparse.ArgumentParser(description="Configure the project")
  parser.add_argument("--repo-root",
                      help="Directory containing sources",
                      type=str,
                      default=os.path.abspath(os.path.dirname(__file__)))
  parser.add_argument("--llvm-path", help="Path to llvm-project sources")
  parser.add_argument("--iree-path", help="Path to IREE (used if enabled)")
  parser.add_argument("--no-ccache",
                      help="Disables ccache (if available)",
                      dest="enable_ccache",
                      action="store_false",
                      default=True)
  parser.add_argument("--build-dir",
                      help="Build directory",
                      type=str,
                      default="build")
  return parser.parse_args()


def main(args):
  print(f"-- Python version {sys.version} ({sys.executable})")
  print(f"CCACHE = {args.enable_ccache}")

  llvm_path = None
  llvm_projects = ["sandbox"]
  llvm_configure_args = [
      f"-DLLVM_EXTERNAL_SANDBOX_SOURCE_DIR={args.repo_root}",
  ]

  # Detect IREE (defaults LLVM path as well).
  iree_path = args.iree_path
  if iree_path:
    iree_path = os.path.abspath(iree_path)
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

  # Detect LLVM.
  if args.llvm_path:
    llvm_path = os.path.abspath(args.llvm_path)
    print(f"-- Using explicit llvm-project path: {llvm_path}")
  elif llvm_path:
    print(f"-- Using inferred llvm-project path: {llvm_path}")
  else:
    llvm_path = os.path.join(args.repo_root, "..", "llvm-project")
    print("-- Using default llvm-project path: {llvm_path}")
  if not os.path.exists(llvm_path):
    print(f"ERROR: Could not find llvm-project at {llvm_path}")
    return 1

  # Detect clang.
  clang_path = shutil.which("clang")
  clangpp_path = shutil.which("clang++")
  has_clang = False
  if clang_path and clangpp_path:
    llvm_configure_args.append(f"-DCMAKE_C_COMPILER={clang_path}")
    llvm_configure_args.append(f"-DCMAKE_CXX_COMPILER={clangpp_path}")
    has_clang = True
  else:
    print(
        "WARNING: Could not find clang. Building with default system compiler")

  # Detect lld.
  lld_path = shutil.which("ld.lld")
  if lld_path:
    if has_clang:
      print(f"-- Using LLD: {lld_path}")
      llvm_configure_args.append(f"-DLLVM_USE_LINKER={lld_path}")
    else:
      print("WARNING: Can only use lld with clang")

  # Detect ccache.
  ccache_path = shutil.which("ccache")
  if ccache_path and args.enable_ccache:
    print(f"-- Using ccache")
    llvm_configure_args.append("-DLLVM_CCACHE_BUILD=ON")

  # CMake configure.
  build_dir = os.path.abspath(args.build_dir)
  os.makedirs(build_dir, exist_ok=True)
  cmake_args = [
      "cmake",
      "-GNinja",
      f"-B{build_dir}",
      f"-S{os.path.join(llvm_path, 'llvm')}",
      "-DLLVM_ENABLE_PROJECTS=mlir",
      "-DLLVM_TARGETS_TO_BUILD=X86;NVPTX",
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
  with open(f"{os.path.join(args.repo_root, '.env')}", "wt") as f:
    f.write(f"PYTHONPATH={os.path.join(build_dir, 'tools', 'sandbox', 'python_package')}")

  # Do initial build.
  cmake_args = ["cmake", "--build", build_dir, "--target", "tools/sandbox/all"]
  print(f"-- Performing initial build: {' '.join(cmake_args)}")
  subprocess.check_call(cmake_args, cwd=build_dir)

  return 0


if __name__ == "__main__":
  sys.exit(main(parse_arguments()))
