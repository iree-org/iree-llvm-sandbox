# An experiment for how to do a mondo build of IREE + LLVM + Sandbox where
# we drive the LLVM configuration. This does work, but as can be seen, there is
# a lot of coupling. This produces a build like:
#
#   llvm/
#     bin/
#     tools/
#       sandbox/
#   iree/
#     iree/
#       tools/
#         iree-translate
#
# The advantage (if there is one) is that this puts LLVM primary and lets you
# control all of the knobs.
#
# The other approach is to just let IREE drive the build, pull LLVM in how
# it wants and tell LLVM to also include the sandbox as an external project.
# A minimum incantation to do this is something like (from the sandbox dir):
#
#   cmake -GNinja -S../iree -Bbuild \
#     -DLLVM_EXTERNAL_PROJECTS=sandbox \
#     -DLLVM_EXTERNAL_SANDBOX_SOURCE_DIR=$PWD \
#     -DIREE_BUILD_PYTHON_BINDINGS=ON
#
# This will produce a project with this layout:
#   iree/
#     tools/
#       iree-translate
#   third_party/llvm-project/llvm/
#     bin/
#     tools/
#       sandbox/
#
# As-is now, llvm-project is EXCLUDE_FROM_ALL in this config, so you would
# build the sandbox as:
#   ninja third_party/llvm-project/llvm/tools/sandbox/all
#
# This configuration is presently broken because:
#   - IREE disable LLVM testing (which the sandbox assumes exists) due to
#     a tug-of-war on who provides gtest.
#   - Possibly a couple other nit-picky path things that I have local fixes for.
#
# Also, ergonomically, the deep nesting of the llvm build directory is really
# annoying. We could fix this on the IREE side by forcing the LLVM binary
# directory to llvm/, resulting in a tree like this:
#
#   iree/
#   llvm/
#     llvm/
#       bin/
#       tools/
#         sandbox
#
# Alternatively, we could add an option that let you put the binary dir
# wherever you wanted.
#
# The perfectionist in me wants the first option to work and be good since
# people will someday want to take a source dep on IREE. But I think for
# lowest friction development that is the least likely to break, the second
# option is likely to require the least maintenance. We can add some special
# cases to the IREE build to make it work, and beyond that, it is like you are
# building LLVM but you also get all of the IREE specific options available.
# For a project like sandbox, which wants low overhead, I'm biased to getting
# the second option working. It would be pretty easy to integrate into the
# configure script (basically invoke IREE's cmake instead of LLVM's and tweak
# a couple of options).
#
# Thoughts?
project(IREE_SANDBOX_MONDO)

cmake_minimum_required(VERSION 3.17...3.22)

set(IREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../iree")
set(LLVM_EXTERNAL_MLIR_IREE_DIALECTS_SOURCE_DIR "${IREE_SOURCE_DIR}/llvm-external-projects/iree-dialects")
set(LLVM_EXTERNAL_MLIR_HLO_SOURCE_DIR "${IREE_SOURCE_DIR}/third_party/mlir-hlo")

# Set up enough IREE options to configure LLVM.
include(${IREE_SOURCE_DIR}/build_tools/cmake/iree_cmake_options.cmake)
include(${IREE_SOURCE_DIR}/build_tools/cmake/iree_external_cmake_options.cmake)
iree_set_compiler_cmake_options()
iree_set_llvm_cmake_options()

# Required options.
set(IREE_DISABLE_BUNDLED_GTEST ON)
set(IREE_BUILD_PYTHON_BINDINGS ON)
set(MLIR_ENABLE_BINDINGS_PYTHON ON)
set(MHLO_ENABLE_BINDINGS_PYTHON ON)

# Enable some LLVM features that the IREE config defaults to off.
option(LLVM_INCLUDE_TESTS "" ON)
option(MLIR_INCLUDE_TESTS "" ON)
set(LLVM_INCLUDE_TESTS ON)
set(MLIR_INCLUDE_TESTS ON)

# TODO: Building the public compiler api as a sub-project is not yet supported.
option(IREE_COMPILER_API_STANDALONE "" ON)
set(IREE_COMPILER_API_STANDALONE ON)

# Add the sandbox as an LLVM dep.
iree_add_llvm_external_project(mlir-iree-dialects MLIR_IREE_DIALECTS ${LLVM_EXTERNAL_MLIR_IREE_DIALECTS_SOURCE_DIR})
iree_add_llvm_external_project(mlir-hlo MLIR_HLO ${LLVM_EXTERNAL_MLIR_HLO_SOURCE_DIR})
iree_add_llvm_external_project(sandbox SANDBOX "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# Delegate entirely to the LLVM config.
add_subdirectory("${IREE_SOURCE_DIR}/third_party/llvm-project/llvm" "${CMAKE_CURRENT_BINARY_DIR}/llvm")

# IREE's tablegen rules use a magic MLIR_INCLUDE_DIRS which must be populated.
# TODO: Find a way to not do this.
set(MLIR_INCLUDE_DIRS)
list(APPEND MLIR_INCLUDE_DIRS "${IREE_SOURCE_DIR}/third_party/llvm-project/mlir/include")
list(APPEND MLIR_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/llvm/tools/mlir/include")

# TODO: MHLO does not properly add its interface includes.
function(_hack_llvm_include_paths)
  set(_common_include_dirs
    $<BUILD_INTERFACE:${IREE_SOURCE_DIR}/third_party/mlir-hlo/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/llvm/tools/mlir-hlo/include>
  )
  message(STATUS "***** HACKED INCLUDE: ${_common_include_dirs}")
  # Avoid globally modifying paths by instead adding the include paths to the
  # rules that really should have them in the first place.
  target_include_directories(LLVMSupport PUBLIC ${_common_include_dirs})
  target_include_directories(MLIRSupport PUBLIC ${_common_include_dirs})
endfunction()
_hack_llvm_include_paths()

# There are multiple things that get angry in IREE in an all build.
#   - LLVM's gtest does not configure gmock...
add_subdirectory("${IREE_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/iree" EXCLUDE_FROM_ALL)
