cmake_minimum_required(VERSION 3.20)
project("how-to-optimize-gemm-opencl")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCL)

if(OpenCL_FOUND)
    message(STATUS "OpenCL headers location: ${OpenCL_INCLUDE_DIRS}")
    message(STATUS "OpenCL libraries location: ${OpenCL_LIBRARIES}")
else()
    message(FATAL_ERROR "Cannot find OpenCL")
endif()

if(OpenCL_FOUND)
    add_executable(main main.cpp utils/init_vector.cpp)
    target_link_libraries(main "OpenCL")
endif()
