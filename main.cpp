#include <iostream>
#include <vector>

#include <string>
#include <sstream>
#include <fstream>

#include "settings.hpp"
#include "utils/init_vector.hpp"
#include "utils/timer.hpp"

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include "CL/opencl.hpp"

#define DEBUG 0
#if DEBUG
#define TEST_NUM_REPEATS 1
#endif

std::string readKernel(const std::string &filename) {
    std::ifstream fhpp;
    fhpp.open("settings.hpp");
    std::stringstream hpp_stream;
    hpp_stream << fhpp.rdbuf();
    auto hpp_str = hpp_stream.str();

    std::ifstream f;
    f.open(filename);
    std::stringstream stream;
    stream << f.rdbuf();
    auto kernel_str = stream.str();

    return hpp_str + kernel_str;
}

void RefGemm(const int M, const int N, const int K,
             const float *A,
             const float *B,
             float *C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float c = 0;
            for (int k = 0; k < K; k++) {
                c += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = c;
        }
    }
}

float Compare(const int M, const int N, const float *C, const float *C_ref) {
    float diff, max_diff = 0.0f;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float c = C[i * N + j], c_ref = C_ref[i * N + j];
            diff = std::abs(c - c_ref);
            max_diff = diff > max_diff ? diff : max_diff;
        }
    }
    return max_diff;
}

int worker(cl::Device &device, const int M, const int N, const int K,
           const int test_num_repeats, const std::string kernel_file) {
    // OpenCL context
    auto context = cl::Context(std::vector<cl::Device>{device});

    // OpenCL program
    auto kernels = readKernel(kernel_file);
#if DEBUG
    std::cout << "Kernels script:" << std::endl << kernels << std::endl;
#endif
    cl::Program program(context, kernels);
    if (program.build({device}) != CL_SUCCESS) {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return -1;
    }

    // Prepare data
    std::vector<float> A_host(M * K), B_host(K * N), C_host(M * N, 0.f), C_ref(M * N, 0.f);
    InitVector(A_host);
    InitVector(B_host);

    // Transer host data to OpenCL device buffers
    auto queue = cl::CommandQueue(context, device);
    auto A_device = cl::Buffer(context, CL_MEM_READ_WRITE, A_host.size() * sizeof(float));
    auto B_device = cl::Buffer(context, CL_MEM_READ_WRITE, B_host.size() * sizeof(float));
    auto C_device = cl::Buffer(context, CL_MEM_READ_WRITE, C_host.size() * sizeof(float));
    queue.enqueueWriteBuffer(A_device, CL_TRUE, 0, A_host.size() * sizeof(float), A_host.data());
    queue.enqueueWriteBuffer(B_device, CL_TRUE, 0, B_host.size() * sizeof(float), B_host.data());
    queue.enqueueWriteBuffer(C_device, CL_TRUE, 0, C_host.size() * sizeof(float), C_host.data());

    // OpenCL kernel setup
    auto kernel_name = std::string("GEMM");
    cl::Kernel kernel(program, kernel_name.c_str());
    kernel.setArg(0, M);
    kernel.setArg(1, N);
    kernel.setArg(2, K);
    kernel.setArg(3, A_device);
    kernel.setArg(4, B_device);
    kernel.setArg(5, C_device);

    // Run kernel
    cl::Event event{nullptr};
    auto global_sizes = cl::NullRange,
         local_sizes = cl::NullRange;
    if (kernel_file.find("GEMM0") != std::string::npos) {
        global_sizes = cl::NDRange(M, N);
    } else if (kernel_file.find("GEMM1") != std::string::npos) {
        const size_t tile_size = OCL_GEMM_KERNEL_TILE_SIZE;
        global_sizes = cl::NDRange(M, N);
        local_sizes = cl::NDRange(tile_size, tile_size);
    }
    TickMeter meter;
    meter.Reset();
    for (int i = 0; i < test_num_repeats; i++) {
        meter.Start();

        auto error = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_sizes, local_sizes, nullptr, &event);
        if (error != CL_SUCCESS) {
            std::cout << "error code: " << error << std::endl;
        }
        cl::WaitForEvents({event});

        meter.End();
    }

    // Get data from device to host
    queue.enqueueReadBuffer(C_device, CL_TRUE, 0, C_host.size() * sizeof(float), C_host.data(), nullptr, &event);
    cl::WaitForEvents({event});

    // Compare with Ref
    RefGemm(M, N, K, A_host.data(), B_host.data(), C_ref.data());
    float max_diff = Compare(M, N, C_host.data(), C_ref.data());

    double mean = meter.GetMeanTimeMillisecond(),
           median = meter.GetMedianTimeMillisecond(),
           minimum = meter.GetMinTimeMillisecond();

    double gflops = (2.0 * M * N * K * 1.0e-6) / mean;

    printf("Kernel: %s, M=%d, N=%d, K=%d, mean=%.2fms, median=%.2fms, min=%.2fms, gflops=%f, max_diff=%f\n",
            kernel_file.c_str(), M, N, K, mean, median, minimum, gflops, max_diff);

    return 1;
}

int main(int argc, char **argv) {
    std::string kernel_file = "kernels/GEMM1.cl";
    if (argc == 2) {
        kernel_file = std::string(argv[1]);
    }

    // OpenCL platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
        std::cerr << "Cannot get OpenCL platforms" << std::endl;
        return -1;
    }
    auto platform = platforms[0];
    {
        std::string platform_name;
        platform.getInfo(CL_PLATFORM_NAME, &platform_name);
        std::cout << platform_name << std::endl;
    }

    // OpenCL devices
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.size() == 0) {
        std::cerr << "Cannot get OpenCL devices" << std::endl;
        return -1;
    }
    auto device = devices[0];
    {
        std::string device_name;
        device.getInfo(CL_DEVICE_NAME, &device_name);\
        std::cout << device_name << std::endl;
    }

    const int test_num_repeats = TEST_NUM_REPEATS;
    for (int scale = TEST_SCALE_START; scale <= TEST_SCALE_END; scale *= TEST_SCALE_FACTOR) {
        const int M = OCL_GEMM_M < 0 ? scale : OCL_GEMM_M,
                  N = OCL_GEMM_N < 0 ? scale : OCL_GEMM_N,
                  K = OCL_GEMM_K < 0 ? scale : OCL_GEMM_K;
        worker(device, M, N, K, test_num_repeats, kernel_file);
    }

    return 0;
}
