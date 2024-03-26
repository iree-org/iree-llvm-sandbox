################################################################################
# Substrait
################################################################################

# The way how Substrait uses sanitizers seems to be incompatible with some flags
# set by LLVM, leading to linker errors, so we deactivate them.
set(SUBSTRAIT_CPP_SANITIZE_DEBUG_BUILD OFF)

# In install mode, Abseil creates a target call `check`, which conflicts with an
# LLVM target with the same name. In non-install mode, the target name changes.
set(ABSL_ENABLE_INSTALL OFF)

# LLVM sets this to `ON`, leading to a compilation error due to a file called
# `time.h` in one of Abseil's folders.
set(CMAKE_INCLUDE_CURRENT_DIR OFF)

if(${LLVM_INCLUDE_TESTS})
  message(FATAL_ERROR "LLVM_INCLUDE_TESTS is enabled but breaks the build due "
                      "to a target name clash with 'abseil-cpp'. Please run "
                      "cmake again with '-DLLVM_INCLUDE_TESTS=OFF'.")
endif()

# Add `substrait-cpp` as a subdirectory with above settings.
set(SUBSTRAIT_CPP_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/substrait-cpp)
add_subdirectory(${SUBSTRAIT_CPP_ROOT_DIR})
