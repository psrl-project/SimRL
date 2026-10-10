# dryrun/component/rollout/engine_cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(dryrun_engine_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_engine_cpp
  src/smoke.cc
  bindings.cc
)

target_include_directories(_engine_cpp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(_engine_cpp PRIVATE -O2 -Wall -Wextra -Wpedantic)
