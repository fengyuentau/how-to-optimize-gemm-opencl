#include "opencl_runtime.hpp"

class GemmExecutor {
public:
    GemmExecutor(OpenCLRuntime *_runtime, int _M, int _N, int _K);
    virtual ~GemmExecutor() {};

    void InitializeData(const std::vector<float> &A,
                        const std::vector<float> &B,
                        const std::vector<float> &C);

    virtual void BuildKernel(const std::string &kernel_file, const std::string &kernel_name, size_t maxWorkGroupSize);
    virtual void RunKernel(int test_num_repeats, std::vector<float> &C, std::vector<double> &elapsed_time);

protected:
    OpenCLRuntime *runtime;
    int M;
    int N;
    int K;

    cl::Kernel kernel;
    cl::Buffer A_device;
    cl::Buffer B_device;
    cl::Buffer C_device;
};
