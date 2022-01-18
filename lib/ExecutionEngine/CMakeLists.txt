# This is a copy of the core MLIR AsyncRuntime.cpp which is built with
# hidden symbols that make it hard to use woth python.
# Ideally this should disappear but it is related to deeper destruction of 
# static ojects, dlopen/dlclose/atexit in a multi-threaded environment.
# For now, copy to get off the ground.
add_mlir_library(mlir_async_runtime_copy
SHARED
AsyncRuntime.cpp

EXCLUDE_FROM_LIBMLIR

LINK_LIBS PUBLIC
${LLVM_PTHREAD_LIB}
)
# The following line hides the functions we need in order to load/unload shared
# libraries properly. Comment it to get off the ground.
# set_property(TARGET mlir_async_runtime PROPERTY CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(mlir_async_runtime_copy PRIVATE mlir_async_runtime_EXPORTS)