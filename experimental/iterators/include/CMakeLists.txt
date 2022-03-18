add_subdirectory(Dialects)

add_library(Iterators INTERFACE)
target_include_directories(Iterators INTERFACE .)
target_compile_features(Iterators INTERFACE cxx_std_17)
