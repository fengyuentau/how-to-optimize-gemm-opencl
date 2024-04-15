#include "gemm_executor.hpp"

GemmExecutor::GemmExecutor(OpenCLRuntime *_runtime, int _M, int _N, int _K)
     : runtime(_runtime), M(_M), N(_N), K(_K) {

        // kernel = nullptr;
        // A_device = nullptr;
        // B_device = nullptr;
        // C_device = nullptr;
    }

void GemmExecutor::InitializeData(const std::vector<float> &A,
                                  const std::vector<float> &B,
                                  const std::vector<float> &C) {
    A_device = cl::Buffer(runtime->GetContext(), CL_MEM_READ_WRITE, A.size() * sizeof(float));
    B_device = cl::Buffer(runtime->GetContext(), CL_MEM_READ_WRITE, B.size() * sizeof(float));
    C_device = cl::Buffer(runtime->GetContext(), CL_MEM_READ_WRITE, C.size() * sizeof(float));
    runtime->GetCommandQueue().enqueueWriteBuffer(A_device, CL_TRUE, 0, A.size() * sizeof(float), A.data());
    runtime->GetCommandQueue().enqueueWriteBuffer(B_device, CL_TRUE, 0, B.size() * sizeof(float), B.data());
    runtime->GetCommandQueue().enqueueWriteBuffer(C_device, CL_TRUE, 0, C.size() * sizeof(float), C.data());
}
