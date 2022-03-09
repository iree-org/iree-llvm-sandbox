add_library(runtime_utils
  SHARED
  RuntimeUtils.cpp
)
set_property(TARGET runtime_utils PROPERTY CXX_STANDARD 11)
target_compile_definitions(runtime_utils PRIVATE runtime_utils_EXPORTS)
