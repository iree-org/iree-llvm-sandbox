cmake_minimum_required(VERSION 3.14)
project(database-iterators-example)
enable_testing()

# Use C++17. Required for some template magic.
set(CMAKE_CXX_STANDARD 17)

# Set up googletest as per its quickstart guide.
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Add subdirectories.
add_subdirectory(docs)
add_subdirectory(include)
add_subdirectory(unittests)
