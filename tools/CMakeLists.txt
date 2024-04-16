cmake_minimum_required(VERSION 3.0)
project(cublas_timing)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architecture (optional)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61) # Example for CUDA 6.1

# Include and link CUDA
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64) # Assuming CUDA is installed in the default location

# Add executable
cuda_add_executable(cublas_timing cublas_timing.cu)

# Link cuBLAS
target_link_libraries(cublas_timing ${CUDA_cublas_LIBRARY})

