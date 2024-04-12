#include <string>
#include <memory>

#include <sstream>
#include <fstream>

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include "CL/opencl.hpp"

#define CHECK_CL_SUCCESS(error, info) \
    if (error != CL_SUCCESS) { \
        printf("OpenCL error code: %d, message: %s\n", int(error), info); \
    }

class OpenCLRuntime {
public:
    OpenCLRuntime() : OpenCLRuntime(0, 0) {}
    OpenCLRuntime(int platform_id, int device_id);

    cl::Kernel BuildKernel(const std::string &kernel_file,
                           const std::string &build_options,
                           const std::string &kernel_name);
    cl::Context &GetContext();
    cl::CommandQueue &GetCommandQueue();

    std::string GetPlatformName() const;
    std::string GetDeviceName() const;
    size_t GetMaxWorkGroupSize() const;

private:
    std::string readKernel(const std::string &kernel_file);

private:
    /*
        -1: platforms.size() <= 0 or platform_id out of range
        -2: devices.size() <= 0 or device_id out of range
        -3: kernels build failed
    */
    int status;

    cl::Platform platform;
    cl::Device device;
    std::shared_ptr<cl::Context> context_ptr;
    std::shared_ptr<cl::CommandQueue> queue_ptr;
};
