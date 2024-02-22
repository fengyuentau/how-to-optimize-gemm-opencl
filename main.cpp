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

bool Compare(const int M, const int N, const float *C, const float *C_ref, const float threshold = 1e-4) {
    bool is_equal = true;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float c = C[i * N + j], c_ref = C_ref[i * N + j];
            if (std::abs(c - c_ref) > threshold) {
                // printf("C[%d][%d] = %f, C_ref[%d][%d] = %f\n", i, j, c, i, j, c_ref);
                is_equal = false;
            }
        }
    }
    return is_equal;
}

int main() {
    // OpenCL platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
        std::cerr << "Cannot get OpenCL platforms" << std::endl;
        return 1;
    }
    auto platform = platforms[0];

    // OpenCL devices
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.size() == 0) {
        std::cerr << "Cannot get OpenCL devices" << std::endl;
        return 1;
    }
    auto device = devices[0];

    // OpenCL context, queue
    auto context = cl::Context(std::vector<cl::Device>{device});
    auto queue = cl::CommandQueue(context, device);

    // OpenCL program
    auto kernels = readKernel("kernels.cl");
    // std::cout << kernels << std::endl;
    cl::Program program(context, kernels);
    if (program.build({device}) != CL_SUCCESS) {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }

    // Prepare data
    const int test_loops = TEST_LOOPS;
    const int M = OCL_GEMM_M, N = OCL_GEMM_N, K = OCL_GEMM_K;
    const int kernel_id = OCL_GEMM_KERNEL;

    auto global_sizes = cl::NDRange(M, N);
    auto local_sizes = cl::NullRange;

    std::vector<float> A_host(M * K), B_host(K * N), C_host(M * N, 0.f), C_ref(M * N, 0.f);
    InitVector(A_host);
    InitVector(B_host);

    // OpenCL buffers
    auto A_device = cl::Buffer(context, CL_MEM_READ_WRITE, A_host.size() * sizeof(float));
    auto B_device = cl::Buffer(context, CL_MEM_READ_WRITE, B_host.size() * sizeof(float));
    auto C_device = cl::Buffer(context, CL_MEM_READ_WRITE, C_host.size() * sizeof(float));
    queue.enqueueWriteBuffer(A_device, CL_TRUE, 0, A_host.size() * sizeof(float), A_host.data());
    queue.enqueueWriteBuffer(B_device, CL_TRUE, 0, B_host.size() * sizeof(float), B_host.data());
    queue.enqueueWriteBuffer(C_device, CL_TRUE, 0, C_host.size() * sizeof(float), C_host.data());

    // OpenCL kernel
    auto kernel_name = std::string("GEMM") + std::to_string(kernel_id);
    cl::Kernel kernel(program, kernel_name.c_str());
    kernel.setArg(0, M);
    kernel.setArg(1, N);
    kernel.setArg(2, K);
    kernel.setArg(3, A_device);
    kernel.setArg(4, B_device);
    kernel.setArg(5, C_device);

    // Run kernel
    TickMeter meter;
    cl::Event event{nullptr};
    for (int i = 0; i < test_loops; i++) {
        meter.Start();

        auto error = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_sizes, local_sizes, nullptr, &event);
        if (error != CL_SUCCESS) {
            std::cout << "error code: " << error << std::endl;
        }
        cl::WaitForEvents({event});

        meter.End();
    }

    // Send back data from device to host
    queue.enqueueReadBuffer(C_device, CL_TRUE, 0, C_host.size() * sizeof(float), C_host.data(), nullptr, &event);
    cl::WaitForEvents({event});

    // Compare with Ref
    RefGemm(M, N, K, A_host.data(), B_host.data(), C_ref.data());
    bool ret = Compare(M, N, C_host.data(), C_ref.data());
    if (ret) {
        std::cout << "Acc: All good!" << std::endl;
    } else {
        std::cout << "Acc: Something is wrong!" << std::endl;
    }

    // Print benchmark results
    double mean = meter.GetMeanTimeMillisecond(),
           median = meter.GetMedianTimeMillisecond(),
           minimum = meter.GetMinTimeMillisecond();
    printf("mean=%.2fms, median=%.2fms, min=%.2fms\n",
           mean, median, minimum);

    return 0;
}
