cmake_minimum_required(VERSION 3.10)
project(matrix-lib-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

include_directories(include)

# Test executable
add_executable(matrix-test tests/main.cpp)

add_executable(matrix-demo examples/demo.cpp)

add_executable(matrix-benchmark examples/benchmark.cpp)

