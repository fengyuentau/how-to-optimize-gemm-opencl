#include "opencv_gemm_executor.hpp"

OpenCVGemmExecutor::OpenCVGemmExecutor(OpenCLRuntime *_runtime, int _M, int _N, int _K)
 : GemmExecutor(_runtime, _M, _N, _K) {
    int vectorWidths[] = { 4, 4, 2, 2, 1, 4, cn, -1 };
    int matWidths[] = {K, N}; // B.width, C.width
    kercn = checkOptimalVectorWidth(vectorWidths, matWidths); // checkOptimalVectorWidth
}

void OpenCVGemmExecutor::BuildKernel(const std::string &kernel_file, const std::string &kernel_name, size_t maxWorkGroupSize) {
    int output_size_min = std::min(M, N);
    int wg_size = std::min(int(maxWorkGroupSize), output_size_min * output_size_min);
    block_size = (wg_size / (32*cn) < 32) ? (wg_size / (16*cn) < 16) ? (wg_size / (8*cn) < 8) ? 1 : 8 : 16 : 32;

    std::string build_options;
    build_options = "-DT=float -DT1=float -DWT=float4 -Dcn=" + std::to_string(cn) + " -Dkercn=" + std::to_string(kercn) + " -DLOCAL_SIZE=" + std::to_string(block_size);
    build_options += (N % block_size != 0) ? " -D NO_MULT" : "";
    // build_options += ""; // haveC
    // build_options += ""; // doubleSupport

    kernel = runtime->BuildKernel(kernel_file, build_options, kernel_name);
}

void OpenCVGemmExecutor::RunKernel(int test_num_repeats, std::vector<float> &C, std::vector<double> &elapsed_time) {
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

    auto global_sizes = cl::NDRange(size_t(N * cn / kercn), size_t(M));
    auto local_sizes = cl::NDRange(size_t(block_size), size_t(block_size));

    cl::Event event{nullptr};
    for (int i = 0; i < test_num_repeats; i++) {
        auto error = runtime->GetCommandQueue().enqueueNDRangeKernel(kernel, cl::NullRange, global_sizes, local_sizes, nullptr, &event);
        if (error != CL_SUCCESS) {
            std::cout << "error code: " << error << std::endl;
        }
        cl::WaitForEvents({event});
        double start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        double end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        elapsed_time.push_back(((end - start) / 1000.0));
    }

    // Get data from device to host
    runtime->GetCommandQueue().enqueueReadBuffer(C_device, CL_TRUE, 0, C.size() * sizeof(float), C.data(), nullptr, &event);
    cl::WaitForEvents({event});
}

int OpenCVGemmExecutor::checkOptimalVectorWidth(const int *vectorWidths, const int *matWidths) {
    int ckercn = vectorWidths[depth];
    std::vector<size_t> offsets{0, 0},
                        steps{size_t(matWidths[0]) * 4, size_t(matWidths[1]) * 4},
                        cols{size_t(matWidths[0]), size_t(matWidths[1])};
    std::vector<int> dividers{ckercn * 4, ckercn * 4},
                        kercns{ckercn, ckercn};

    size_t size = offsets.size();

    for (size_t i = 0; i < size; ++i)
        while (offsets[i] % dividers[i] != 0 || steps[i] % dividers[i] != 0 || cols[i] % kercns[i] != 0)
            dividers[i] >>= 1, kercns[i] >>= 1;

    // default strategy
    int kercn = *std::min_element(kercns.begin(), kercns.end());
    return kercn;
}
