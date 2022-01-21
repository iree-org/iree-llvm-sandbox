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
  parser.add_argument(
      "--target",
      help="Semicolumn-separated list of targets to build with LLVM",
      default="X86;NVPTX")
  parser.add_argument("--build-dir",
                      help="Build directory",
                      type=str,
                      default="build")
  parser.add_argument("--build-mode",
                      help="Build mode (Release, Debug or RelWithDebInfo)",
                      type=str,
                      default="Release")
  # Boolean flags: all deactivated by default
  # Activate with e.g. --lld.
  # Also supports e.g. --no-lld.
  parser.add_argument(
      "--lld",
      help="Build with ENABLE_LLD=ON (optional)",
      dest="enable_lld",
      default=True,
      action="store_true",
  )
  parser.add_argument(
      "--asan",
      help="Build with LLVM_USE_SANITIZER=Address (optional)",
      dest="enable_asan",
      default=False,
      action="store_true",
  )
  parser.add_argument(
      "--alp",
      help="Build with SANDBOX_ENABLE_ALP=ON (optional)",
      dest="enable_alp",
      default=False,
      action="store_true",
  )
  parser.add_argument(
      "--ccache",
      help="Enable ccache (if available)",
      dest="enable_ccache",
      default=True,
      action="store_true",
  )
  parser.add_argument(
      "--use-system-cc",
      help="Use the default system compiler" +
      "\n[warning] Setting to false seems to trigger spurious rebuilds",
      dest="enable_system_cc",
      default=False,
      action="store_true",
  )
  parser.add_argument(
      "--cuda-runner",
      help="Build cuda runner library",
      dest="cuda_runner",
      default=False,
      action=argparse.BooleanOptionalAction,
  )
  return parser.parse_args()


def read_through_symlink(link):
  p = subprocess.Popen(['readlink', '-f', link], stdout=subprocess.PIPE)
  p.wait()
  return p.communicate()[0].rstrip().decode("utf-8")


def main(args):
  print(f"-- Python version {sys.version} ({sys.executable})")
  print(f"CCACHE = {args.enable_ccache}")

  llvm_path = None
  llvm_projects = ["sandbox"]
  llvm_configure_args = [
      f"-DLLVM_EXTERNAL_SANDBOX_SOURCE_DIR={args.repo_root}",
  ]

  # TODO: Make configurable.
  llvm_builtin_projects = ["mlir", "clang", "clang-tools-extra"]

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
    print(f"-- Using default llvm-project path: {llvm_path}")
  if not os.path.exists(llvm_path):
    print(f"ERROR: Could not find llvm-project at {llvm_path}")
    return 1

  # Detect clang.
  if not bool(args.enable_system_cc):
    clang_path = shutil.which("clang")
    clangpp_path = shutil.which("clang++")
    if clang_path and clangpp_path:
      # Adding DCMAKE_C_COMPILER / DCMAKE_CXX_COMPILER to symlinks triggers
      # spurious rebuilds everywhere.
      # Instead, read through the symlinks.
      # TOD: Reenable because atm this still fails with:
      # -- Check for working C compiler: /usr/lib/llvm-11/bin/clang - skipped
      # -- Check for working CXX compiler: /usr/lib/llvm-11/bin/clang - skipped
      # -- Performing Test LLVM_LIBSTDCXX_MIN
      # -- Performing Test LLVM_LIBSTDCXX_MIN - Failed
      # CMake Error at cmake/modules/CheckCompilerVersion.cmake:97 (message):
      #   libstdc++ version must be at least 5.1.
      # Call Stack (most recent call first):
      #   cmake/config-ix.cmake:14 (include)
      #   CMakeLists.txt:726 (include)
      # clang_path = read_through_symlink(clang_path)
      # clangpp_path = read_through_symlink(clangpp_path)
      llvm_configure_args.append(f"-DCMAKE_C_COMPILER={clang_path}")
      llvm_configure_args.append(f"-DCMAKE_CXX_COMPILER={clangpp_path}")
      print("-- Building with clang")
    else:
      print(
          "WARNING: Could not find clang. Building with default system compiler"
      )
  else:
    print("-- Building with default system compiler")

  # Detect lld.
  if args.enable_lld:
    lld_path = shutil.which("ld.lld")
    if lld_path:
      print(f"-- Using lld: {lld_path}")
      llvm_configure_args.append("-DLLVM_ENABLE_LLD=ON")
    else:
      print("WARNING: LLD (ld.lld) not found on path. Configure may fail.")

  # Optionally enable Alp
  if args.enable_alp:
    llvm_configure_args.append("-DSANDBOX_ENABLE_ALP=ON")

  # Detect ccache.
  if args.enable_ccache:
    ccache_path = shutil.which("ccache")
    if ccache_path:
      print(f"-- Using ccache: {ccache_path}")
      llvm_configure_args.append("-DLLVM_CCACHE_BUILD=ON")
    else:
      print("WARNING: Project developers use ccache which is not installed")
  if args.cuda_runner:
    llvm_configure_args.append("-DMLIR_ENABLE_CUDA_RUNNER=ON")
  # CMake configure.
  build_dir = os.path.abspath(args.build_dir)
  build_mode = args.build_mode
  os.makedirs(build_dir, exist_ok=True)
  if args.enable_asan:
    llvm_configure_args.append("-DLLVM_USE_SANITIZER=Address")
  cmake_args = [
      "cmake",
      "-GNinja",
      f"-B{build_dir}",
      f"-S{os.path.join(llvm_path, 'llvm')}",
      f"-DLLVM_ENABLE_PROJECTS={';'.join(llvm_builtin_projects)}",
      f"-DLLVM_TARGETS_TO_BUILD={args.target}",
      "-DMLIR_INCLUDE_INTEGRATION_TESTS=ON",
      "-DLLVM_ENABLE_ASSERTIONS=ON",
      "-DBUILD_SHARED_LIBS=ON",
      "-DLLVM_INCLUDE_UTILS=ON",
      "-DLLVM_INSTALL_UTILS=ON",
      "-DLLVM_BUILD_EXAMPLES=ON",
      "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
      f"-DPython3_EXECUTABLE={sys.executable}",
      f"-DCMAKE_BUILD_TYPE={build_mode}",
      f"-DLLVM_EXTERNAL_PROJECTS={';'.join(llvm_projects)}",
  ] + llvm_configure_args
  print(f"-- Running cmake:\n  {' '.join(cmake_args)}")
  subprocess.check_call(cmake_args, cwd=build_dir)

  # Write out .env.
  with open(f"{os.path.join(args.repo_root, '.env')}", "wt") as f:
    f.write(
        f"PYTHONPATH={os.path.join(build_dir, 'tools', 'sandbox', 'python_package')}"
    )

  # Do initial build.
  # Also build all the relevant tools to disassemble and run analyses.
  cmake_args = ["cmake", "--build", build_dir, "--target", \
                "tools/sandbox/all", "mlir-opt", "mlir-translate", \
                "mlir-cpu-runner", "mlir_runner_utils", "mlir_c_runner_utils", \
                "mlir_async_runtime_copy", "llvm-mca", "llvm-objdump", "llc", "opt", \
                "FileCheck"]

  if args.enable_alp:
    cmake_args.append("clang")
    cmake_args.append("clang-cpp")

  if args.cuda_runner:
    cmake_args.append("mlir_cuda_runtime")
    
  print(f"-- Performing initial build: {' '.join(cmake_args)}")
  subprocess.check_call(cmake_args, cwd=build_dir)

  return 0


if __name__ == "__main__":
  sys.exit(main(parse_arguments()))
