cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(quadruped_cpp)

find_package(Torch REQUIRED)

add_executable(load_model load_model.cpp)
target_link_libraries(load_model "${TORCH_LIBRARIES}")
set_property(TARGET load_model PROPERTY CXX_STANDARD 14)
