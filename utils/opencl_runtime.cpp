#include "opencl_runtime.hpp"

OpenCLRuntime::OpenCLRuntime(int platform_id, int device_id) {
    status = 1;

    std::vector<cl::Platform> platforms;
    auto err = cl::Platform::get(&platforms);
    CHECK_CL_SUCCESS(err, "Get platforms");
    if (platforms.size() > 0 && platforms.size() > platform_id) {
        platform = platforms[platform_id];
    } else {
        status = -1;
    }

    std::vector<cl::Device> devices;
    err = platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    CHECK_CL_SUCCESS(err, "Get devices");
    if (devices.size() > 0 && devices.size() > device_id) {
        device = devices[device_id];
    } else {
        status = -2;
    }

    device.getInfo(CL_DEVICE_EXTENSIONS, &extentions);

    context_ptr = std::make_shared<cl::Context>(std::vector<cl::Device>{device});
    queue_ptr = std::make_shared<cl::CommandQueue>(*context_ptr, device, CL_QUEUE_PROFILING_ENABLE);
}

cl::Kernel OpenCLRuntime::BuildKernel(const std::string &kernel_file,
                                      const std::string &build_options,
                                      const std::string &kernel_name) {
    auto kernel_str = readKernel(kernel_file);

    cl::Program program(*context_ptr, kernel_str);
    auto err = program.build({device}, build_options.c_str());
    CHECK_CL_SUCCESS(err, "Build kernel");
    if (err != CL_SUCCESS) {
        status = -3;
    }

    cl::Kernel kernel(program, kernel_name.c_str());
    return kernel;
}

cl::Context &OpenCLRuntime::GetContext() {
    return *context_ptr;
}

cl::CommandQueue &OpenCLRuntime::GetCommandQueue() {
    return *queue_ptr;
}

std::string OpenCLRuntime::GetPlatformName() const {
    std::string name;
    platform.getInfo(CL_PLATFORM_NAME, &name);
    return name;
}

std::string OpenCLRuntime::GetDeviceName() const {
    std::string name;
    device.getInfo(CL_DEVICE_NAME, &name);
    return name;
}

size_t OpenCLRuntime::GetMaxWorkGroupSize() const {
    size_t MaxWorkGroupSize;
    device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &MaxWorkGroupSize);
    return MaxWorkGroupSize;
}

bool OpenCLRuntime::IsIntelSubgroupSupported()const {
    return extentions.find("cl_intel_subgroups") != std::string::npos;
}

std::string OpenCLRuntime::readKernel(const std::string &kernel_file) {
    std::ifstream fhpp;
    fhpp.open("settings.hpp");
    std::stringstream hpp_stream;
    hpp_stream << fhpp.rdbuf();
    auto hpp_str = hpp_stream.str();

    std::ifstream f;
    f.open(kernel_file);
    std::stringstream stream;
    stream << f.rdbuf();
    auto kernel_str = stream.str();

    return hpp_str + kernel_str;
}
