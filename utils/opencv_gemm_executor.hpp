#include "gemm_executor.hpp"
#include <algorithm>
#include <iostream>

class OpenCVGemmExecutor : public GemmExecutor {
public:
    OpenCVGemmExecutor(OpenCLRuntime *_runtime, int _M, int _N, int _K);

    void BuildKernel(const std::string &kernel_file, const std::string &kernel_name, size_t maxWorkGroupSize) override;
    void RunKernel(int test_num_repeats, std::vector<float> &C, std::vector<double> &elapsed_time) override;

private:
    int type{5};    // CV_32F=5
    int depth{5};   // CV_MAT_DEPTH(CV_32F)=5
    int cn{1};      // CV_MAT_CN(CV_32F)=1
    int kercn;
    int block_size;

private:
    int checkOptimalVectorWidth(const int *vectorWidths, const int *matWidths);
};
