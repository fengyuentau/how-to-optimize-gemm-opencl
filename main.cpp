#include <iostream>
#include <vector>
#include <random>

#include <string>
#include <sstream>
#include <fstream>

#include "settings.hpp"

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

void initVector(std::vector<float> &vec) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    for (int i = 0; i < vec.size(); i++) {
        vec[i] = distribution(gen);
    }
}

void initVector(std::vector<float> &vec, float val) {
    for (int i = 0; i < vec.size(); i++) {
        vec[i] = val;
    }
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
    for (size_t i = 0; i < platforms.size(); i++) {
        std::string platform_name;
        auto status = platforms[i].getInfo(CL_PLATFORM_NAME, &platform_name);
        std::cout << platform_name << std::endl;
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
    // size_t maxWorkGroupSize; // Max number of work iterms allowed in a group
    // device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);
    // std::cout << "Maximum Work Group Size: " << maxWorkGroupSize << std::endl;


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

    // /* Get PTX assembly and save
    // */
    // // Get the program binary for the device
    // std::vector<unsigned char> binary = program.getInfo<CL_PROGRAM_BINARIES>()[0];
    // // Write the binary to a file
    // std::ofstream binaryFile("kernel" + std::to_string(OCL_GEMM_KERNEL) + ".bin", std::ios::out | std::ios::binary);
    // binaryFile.write(reinterpret_cast<char*>(binary.data()), binary.size());
    // binaryFile.close();

    // Prepare data
    const int M = OCL_GEMM_M, N = OCL_GEMM_N, K = OCL_GEMM_K;
    const int kernel_id = OCL_GEMM_KERNEL;

    auto global_sizes = cl::NDRange(M, N);
    auto local_sizes = cl::NullRange;

    std::vector<float> A_host(M * K), B_host(K * N), C_host(M * N, 0.f), C_ref(M * N, 0.f);
    initVector(A_host);
    initVector(B_host);
    initVector(C_host, 0.f);

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

    auto error = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_sizes, local_sizes);
    if (error != CL_SUCCESS) {
        std::cout << "error code: " << error << std::endl;
    }

    // Send back data from device to host
    cl::finish();
    queue.enqueueReadBuffer(C_device, CL_TRUE, 0, C_host.size() * sizeof(float), C_host.data());

    // Compare with Ref
    RefGemm(M, N, K, A_host.data(), B_host.data(), C_ref.data());
    bool ret = Compare(M, N, C_host.data(), C_ref.data());
    if (ret) {
        std::cout << "All good!" << std::endl;
    } else {
        std::cout << "Something is wrong!" << std::endl;
    }

    return 0;
}
