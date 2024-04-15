#include <iostream>
#include <vector>
#include <algorithm>

#include <string>
#include <sstream>
#include <fstream>

#include "settings.hpp"
#include "utils/init_vector.hpp"
#include "utils/opencl_runtime.hpp"

#ifdef HAVE_CLBLAST
#include "clblast.h"
#endif

#define DEBUG 0
#if DEBUG
#define TEST_NUM_REPEATS 1
#endif

double GetMeanTimeMillisecond(std::vector<double> elapsed_times) {
    return std::accumulate(elapsed_times.begin(), elapsed_times.end(), 0.f) / (double)elapsed_times.size();
}

double GetMedianTimeMillisecond(std::vector<double> elapsed_times) {
    std::sort(elapsed_times.begin(), elapsed_times.end());

    auto half_length = static_cast<size_t>(elapsed_times.size() / 2);
    return elapsed_times.size() % 2 == 0 ?
                (elapsed_times[half_length - 1] + elapsed_times[half_length]) / 2.0f :
                elapsed_times[half_length];
}

double GetMinTimeMillisecond(std::vector<double> elapsed_times) {
    std::sort(elapsed_times.begin(), elapsed_times.end());

    return elapsed_times.front();
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

int worker(OpenCLRuntime& runtime, const int M, const int N, const int K,
           const int test_num_repeats, const std::string &kernel_file) {
    std::string build_options, kernel_name("GEMM");
    if (kernel_file.find("opencv_gemm") != std::string::npos) {
        size_t maxWorkGroupSize = runtime.GetMaxWorkGroupSize();
        int output_size_min = std::min(M, N);
        int wg_size = std::min(int(maxWorkGroupSize), output_size_min * output_size_min);
        int cn = 1;
        int kercn = 4;
        int block_size = (wg_size / (32*cn) < 32) ? (wg_size / (16*cn) < 16) ? (wg_size / (8*cn) < 8) ? 1 : 8 : 16 : 32;

        build_options = "-DT=float -DT1=float -DWT=float4 -Dcn=" + std::to_string(cn) + " -Dkercn=" + std::to_string(kercn) + " -DLOCAL_SIZE=" + std::to_string(block_size);
        build_options += (N % block_size != 0) ? " -D NO_MULT" : "";
        // build_options += ""; // haveC
        // build_options += ""; // doubleSupport

        kernel_name = "gemm";
    } else if (kernel_file.find("opencv_intel_gemm") != std::string::npos) {
        if (M % 32 == 0 && N % 32 == 0 && K % 16 == 0) {
            kernel_name = "intelblas_gemm_buffer_NN_sp";
        } else {
            // if (M % 2 != 0)
            //     return false;
            // // vload4(0, dst_write0) - 4 cols
            // // multiply by lx: 8
            // if (N % (4*8) != 0)
            //     return false;
            kernel_name = "intelblas_gemm_buffer_NN";
        }
    } else if (kernel_file.find("opencv_dnn_gemm_buffer") != std::string::npos) { // this kernel is normally run at half precision
        build_options = "-DTYPE=1 -DZERO_BETA=1";

        kernel_name = "gemm_buffer_NN_float";
    } else if (kernel_file.find("mnn") != std::string::npos) {
        build_options = "-DFLOAT=float  -DFLOAT2=float2 -DFLOAT3=float3 -DFLOAT4=float4 -DFLOAT8=float8 -DRI_F=read_imagef -DFLOAT16=float16 -DWI_F=write_imagef -DCONVERT_FLOAT4=convert_float4 -DCONVERT_FLOAT8=convert_float8 -DCONVERT_FLOAT16=convert_float16";

        kernel_name = "matmul_buf";
    }
    auto kernel = runtime.BuildKernel(kernel_file, build_options, kernel_name);

    // Prepare data
    std::vector<float> A_host(M * K), B_host(K * N), C_host(M * N, 0.f), C_ref(M * N, 0.f);
    InitVector(A_host);
    InitVector(B_host);

    // Transer host data to OpenCL device buffers
    auto A_device = cl::Buffer(runtime.GetContext(), CL_MEM_READ_WRITE, A_host.size() * sizeof(float));
    auto B_device = cl::Buffer(runtime.GetContext(), CL_MEM_READ_WRITE, B_host.size() * sizeof(float));
    auto C_device = cl::Buffer(runtime.GetContext(), CL_MEM_READ_WRITE, C_host.size() * sizeof(float));
    runtime.GetCommandQueue().enqueueWriteBuffer(A_device, CL_TRUE, 0, A_host.size() * sizeof(float), A_host.data());
    runtime.GetCommandQueue().enqueueWriteBuffer(B_device, CL_TRUE, 0, B_host.size() * sizeof(float), B_host.data());
    runtime.GetCommandQueue().enqueueWriteBuffer(C_device, CL_TRUE, 0, C_host.size() * sizeof(float), C_host.data());

    // OpenCL kernel setup
    auto global_sizes = cl::NullRange,
         local_sizes = cl::NullRange;
    if (kernel_file.find("opencv_gemm") != std::string::npos) {
        size_t maxWorkGroupSize = runtime.GetMaxWorkGroupSize();
        int output_size_min = std::min(M, N);
        int wg_size = std::min(int(maxWorkGroupSize), output_size_min * output_size_min);
        int cn = 1;
        int kercn = 4;
        int block_size = (wg_size / (32*cn) < 32) ? (wg_size / (16*cn) < 16) ? (wg_size / (8*cn) < 8) ? 1 : 8 : 16 : 32;

        const int A_step = K * 4, A_offset = 0,
                  B_step = N * 4, B_offset = 0,
                  C_step = N * 4, C_offset = 0,
                  C_rows = M, C_cols = N * cn / kercn; // kercn, cn
        const int width = K;
        kernel.setArg(0, A_device);
        kernel.setArg(1, A_step);
        kernel.setArg(2, A_offset);

        kernel.setArg(3, B_device);
        kernel.setArg(4, B_step);
        kernel.setArg(5, B_offset);

        kernel.setArg(6, C_device);
        kernel.setArg(7, C_step);
        kernel.setArg(8, C_offset);
        kernel.setArg(9, C_rows);
        kernel.setArg(10, C_cols);

        kernel.setArg(11, width);
        kernel.setArg(12, 1.f);
        kernel.setArg(13, 0.f);

        // global size
        global_sizes = cl::NDRange(size_t(N * cn / kercn), size_t(M));
        // local size
        local_sizes = cl::NDRange(size_t(block_size), size_t(block_size));

    } else if (kernel_file.find("opencv_intel_gemm") != std::string::npos) {
        unsigned int lx = 8, ly = 4;
        unsigned int dx = 4, dy = 8;

        // divUp: (a + b - 1) / b
        auto divUp = [](size_t a, size_t b) {
            return (a + b - 1) / b;
        };
        const size_t gx = divUp((size_t)N, dx);
        const size_t gy = divUp((size_t)M, dy);

        // size_t local[] = {lx, ly, 1};
        local_sizes = cl::NDRange(lx, ly, 1);
        // roundUp: a + b - 1 - (a + b -1) % b
        auto roundUp = [](size_t a, size_t b) {
            return a + b - 1 - (a + b -1) % b;
        };
        // size_t global[] = {roundUp(gx, lx), roundUp(gy, ly), 1};
        global_sizes = cl::NDRange(roundUp(gx, lx), roundUp(gy, ly), 1);

    } else if (kernel_file.find("opencv_dnn_gemm_buffer") != std::string::npos) {
        size_t sub_group_size = 8; // float16: 16

        size_t lx = sub_group_size;
        // size_t ly = (TransB != CblasNoTrans && TransA == CblasNoTrans && halfPrecisionMode) ? 2 : 4;
        size_t ly = 4;
        // int dx = (TransB != CblasNoTrans && TransA == CblasNoTrans) ? 1 : 4;
        int dx = 4;
        int dy = 8;
        size_t gx = (size_t)(N + dx - 1) / dx;
        size_t gy = (size_t)(M + dy - 1) / dy;

        global_sizes = cl::NDRange((gx + lx - 1) / lx * lx, (gy + ly - 1) / ly * ly);
        local_sizes = cl::NDRange(lx, ly);
    } else if (kernel_file.find("mnn") != std::string::npos) {
        const int height = M,
                  outputChannel = N,
                  width = N,
                  outputChannelBlocks = static_cast<int>((outputChannel + 4 - 1) / 4),
                  widthBlocks = static_cast<int>((width + 4 - 1) / 4);
        kernel.setArg(0, widthBlocks);
        kernel.setArg(1, height);
        kernel.setArg(2, A_device);
        kernel.setArg(3, B_device);
        kernel.setArg(4, C_device); // no_bias
        kernel.setArg(5, outputChannel);
        kernel.setArg(6, outputChannelBlocks);
        kernel.setArg(7, widthBlocks);
        kernel.setArg(8, width);

        global_sizes = cl::NDRange(M, N);
    } else { // GEMM
        kernel.setArg(0, M);
        kernel.setArg(1, N);
        kernel.setArg(2, K);
        kernel.setArg(3, A_device);
        kernel.setArg(4, B_device);
        kernel.setArg(5, C_device);

        if (kernel_file.find("GEMM0") != std::string::npos) {
            global_sizes = cl::NDRange(M, N);
        } else if (kernel_file.find("GEMM1") != std::string::npos) {
            const size_t tile_size = OCL_GEMM_KERNEL_TILE_SIZE;
            global_sizes = cl::NDRange(M, N);
            local_sizes = cl::NDRange(tile_size, tile_size); // tile_size x tile_size <= Max work group size
        } else if (kernel_file.find("GEMM2") != std::string::npos) {
            const size_t tile_size = OCL_GEMM_KERNEL_TILE_SIZE;
            const size_t work_per_thread = OCL_GEMM_KERNEL_WORK_PER_THREAD;
            global_sizes = cl::NDRange(M, N / work_per_thread);
            local_sizes = cl::NDRange(tile_size, tile_size / work_per_thread);
        }
    }

    // Run kernel
    cl::Event event{nullptr};
    std::vector<double> elapsed_times;
    if (kernel_file.find("opencv_intel_gemm") != std::string::npos) {
        int stride = (M * N < 1024 * 1024) ? 10000000 : 256;
        for (int i = 0; i < test_num_repeats; i++) {
            double elapsed_time = 0;
            for(int start_index = 0; start_index < K; start_index += stride) {
                kernel.setArg(0, A_device);
                kernel.setArg(1, 0);
                kernel.setArg(2, B_device);
                kernel.setArg(3, 0);
                kernel.setArg(4, C_device);
                kernel.setArg(5, 0);
                kernel.setArg(6, M);
                kernel.setArg(7, N);
                kernel.setArg(8, K);
                kernel.setArg(9, 1.0f);
                kernel.setArg(10, 0.f);
                kernel.setArg(11, K);
                kernel.setArg(12, N);
                kernel.setArg(13, N);
                kernel.setArg(14, start_index);
                kernel.setArg(15, stride);

                auto error = runtime.GetCommandQueue().enqueueNDRangeKernel(kernel, cl::NullRange, global_sizes, local_sizes, nullptr, &event);
                if (error != CL_SUCCESS) {
                    std::cout << "error code: " << error << std::endl;
                }
                cl::WaitForEvents({event});
                double start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
                double end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
                elapsed_time += (end - start) / 1000.0;
            }
            elapsed_times.push_back(elapsed_time);
        }
    } else if (kernel_file.find("opencv_dnn_gemm_buffer") != std::string::npos) {
        int stride = 256;
        for (int i = 0; i < test_num_repeats; i++) {
            double elapsed_time = 0;
            for(int start_index = 0; start_index < K; start_index += stride) {
                kernel.setArg(0, A_device);
                kernel.setArg(1, 0);
                kernel.setArg(2, B_device);
                kernel.setArg(3, 0);
                kernel.setArg(4, C_device);
                kernel.setArg(5, 0);
                kernel.setArg(6, M);
                kernel.setArg(7, N);
                kernel.setArg(8, K);
                kernel.setArg(9, 1.0f);
                kernel.setArg(10, 0.f);
                kernel.setArg(11, start_index);

                auto error = runtime.GetCommandQueue().enqueueNDRangeKernel(kernel, cl::NullRange, global_sizes, local_sizes, nullptr, &event);
                if (error != CL_SUCCESS) {
                    std::cout << "error code: " << error << std::endl;
                }
                cl::WaitForEvents({event});
                double start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
                double end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
                elapsed_time += (end - start) / 1000.0;
            }
            elapsed_times.push_back(elapsed_time);
        }
    } else {
        for (int i = 0; i < test_num_repeats; i++) {
            auto error = runtime.GetCommandQueue().enqueueNDRangeKernel(kernel, cl::NullRange, global_sizes, local_sizes, nullptr, &event);
            if (error != CL_SUCCESS) {
                std::cout << "error code: " << error << std::endl;
            }
            cl::WaitForEvents({event});
            double start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            double end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            elapsed_times.push_back(((end - start) / 1000.0));
        }
    }

    // Get data from device to host
    runtime.GetCommandQueue().enqueueReadBuffer(C_device, CL_TRUE, 0, C_host.size() * sizeof(float), C_host.data(), nullptr, &event);
    cl::WaitForEvents({event});

    // Compare with Ref
    float max_diff = 0.f;
    if (M < 1024) {
        RefGemm(M, N, K, A_host.data(), B_host.data(), C_ref.data());
    }
    max_diff = Compare(M, N, C_host.data(), C_ref.data());

    double mean = GetMeanTimeMillisecond(elapsed_times),
           median = GetMedianTimeMillisecond(elapsed_times),
           minimum = GetMinTimeMillisecond(elapsed_times);

    double gflops = (2.0 * M * N * K * 1.0e-6) / mean;

    printf("Kernel: %s, M=%d, N=%d, K=%d, mean=%.2fms, median=%.2fms, min=%.2fms, gflops=%f, max_diff=%f\n",
            kernel_file.c_str(), M, N, K, mean, median, minimum, gflops, max_diff);

    return 1;
}

#ifdef HAVE_CLBLAST
int worker_clblast(OpenCLRuntime& runtime, const int M, const int N, const int K,
                   const int test_num_repeats) {
    // Prepare data
    std::vector<float> A_host(M * K), B_host(K * N), C_host(M * N, 0.f), C_ref(M * N, 0.f);
    InitVector(A_host);
    InitVector(B_host);
    int lda = K, ldb = N, ldc = N;

    // Transer host data to OpenCL device buffers
    auto A_device = cl::Buffer(runtime.GetContext(), CL_MEM_READ_WRITE, A_host.size() * sizeof(float));
    auto B_device = cl::Buffer(runtime.GetContext(), CL_MEM_READ_WRITE, B_host.size() * sizeof(float));
    auto C_device = cl::Buffer(runtime.GetContext(), CL_MEM_READ_WRITE, C_host.size() * sizeof(float));
    runtime.GetCommandQueue().enqueueWriteBuffer(A_device, CL_TRUE, 0, A_host.size() * sizeof(float), A_host.data());
    runtime.GetCommandQueue().enqueueWriteBuffer(B_device, CL_TRUE, 0, B_host.size() * sizeof(float), B_host.data());
    runtime.GetCommandQueue().enqueueWriteBuffer(C_device, CL_TRUE, 0, C_host.size() * sizeof(float), C_host.data());

    // Run CLBlast GEMM
    auto queue = runtime.GetCommandQueue();
    auto raw_queue = queue();
    cl::Event event;
    auto raw_event = event();
    std::vector<double> elapsed_times;
    for (int i = 0; i < test_num_repeats; i++) {
        auto status = clblast::Gemm(clblast::Layout::kRowMajor,
                                    clblast::Transpose::kNo,
                                    clblast::Transpose::kNo,
                                    M, N, K,
                                    1.f, // alpha
                                    A_device(), 0, lda,
                                    B_device(), 0, ldb,
                                    0.f, // beta
                                    C_device(), 0, ldc,
                                    &raw_queue, &raw_event);
        if (status == clblast::StatusCode::kSuccess) {
            clWaitForEvents(1, &raw_event);
            cl_ulong start_time, end_time;
            clGetEventProfilingInfo(raw_event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
            clGetEventProfilingInfo(raw_event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
            double elapsed_time = (end_time - start_time) / 1000.0;
            elapsed_times.push_back(elapsed_time);
        }
    }

    // Get data from device to host
    runtime.GetCommandQueue().enqueueReadBuffer(C_device, CL_TRUE, 0, C_host.size() * sizeof(float), C_host.data(), nullptr, &event);
    cl::WaitForEvents({event});

    // Compare with Ref
    float max_diff = 0.f;
    if (M < 1024) {
        RefGemm(M, N, K, A_host.data(), B_host.data(), C_ref.data());
    }
    max_diff = Compare(M, N, C_host.data(), C_ref.data());

    double mean = GetMeanTimeMillisecond(elapsed_times),
           median = GetMedianTimeMillisecond(elapsed_times),
           minimum = GetMinTimeMillisecond(elapsed_times);

    double gflops = (2.0 * M * N * K * 1.0e-6) / mean;

    printf("Kernel: CLBlast, M=%d, N=%d, K=%d, mean=%.2fms, median=%.2fms, min=%.2fms, gflops=%f, max_diff=%f\n",
            M, N, K, mean, median, minimum, gflops, max_diff);

    return 1;
}
#else
int worker_clblast(cl::Device &device, const int M, const int N, const int K,
                const int test_num_repeats) {
    printf("CLBlast not found!\n");
}
#endif

int main(int argc, char **argv) {
    std::string kernel_file = "kernels/GEMM0.cl";
    if (argc == 2) {
        kernel_file = std::string(argv[1]);
    }

    OpenCLRuntime runtime;
    std::cout << runtime.GetPlatformName() << std::endl
              << runtime.GetDeviceName() << std::endl;
    if ((kernel_file.find("intel") != std::string::npos || kernel_file.find("opencv_dnn_gemm_buffer") != std::string::npos)
        && !runtime.IsIntelSubgroupSupported()) {
        printf("Kernel: %s, not supported!\n", kernel_file.c_str());
        return 0;
    }

    const int test_num_repeats = TEST_NUM_REPEATS;
    for (int scale = TEST_SCALE_START; scale <= TEST_SCALE_END; scale *= TEST_SCALE_FACTOR) {
        const int M = OCL_GEMM_M < 0 ? scale : OCL_GEMM_M,
                  N = OCL_GEMM_N < 0 ? scale : OCL_GEMM_N,
                  K = OCL_GEMM_K < 0 ? scale : OCL_GEMM_K;
        if (kernel_file == "clblast") {
            worker_clblast(runtime, M, N, K, test_num_repeats);
        } else {
            worker(runtime, M, N, K, test_num_repeats, kernel_file);
        }
    }

    return 0;
}
