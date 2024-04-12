cmake_minimum_required(VERSION 3.20)
project("how-to-optimize-gemm-opencl")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCL)

if(OpenCL_FOUND)
    message(STATUS "OpenCL headers location: ${OpenCL_INCLUDE_DIRS}")
    message(STATUS "OpenCL libraries location: ${OpenCL_LIBRARIES}")
    # Get opencl.hpp for macOS
    if(APPLE)
        execute_process(
            COMMAND git clone https://github.com/KhronosGroup/OpenCL-CLHPP.git
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            RESULT_VARIABLE clone_result
        )

        if(clone_result EQUAL "0")
            message("KhronosGroup/OpenCL-CLHPP cloned successfully")
            include_directories(${CMAKE_BINARY_DIR}/OpenCL-CLHPP/include)
        else()
            message("Failed to clone KhronosGroup/OpenCL-CLHPP")
        endif()
    endif()
else()
    message(FATAL_ERROR "Cannot find OpenCL")
endif()

if(OpenCL_FOUND)
    add_executable(main main.cpp utils/init_vector.cpp utils/opencl_runtime.cpp)
    if(APPLE)
        target_link_libraries(main "-framework OpenCL")
    else()
        target_link_libraries(main "OpenCL")
    endif()
endif()

option(WITH_CLBLAST "Build with CLBlast" OFF)
if(WITH_CLBLAST)
    find_package(CLBlast)
endif()

if(CLBlast_FOUND)
    message(STATUS "Found CLBlast")
else()
    message(STATUS "Cannot find CLBlast")
endif()

if(CLBlast_FOUND)
    add_compile_definitions(HAVE_CLBLAST)
    target_link_libraries(main "clblast")
endif()
