#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#include "../settings.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

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

void InitVector(std::vector<float> &vec) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    for (int i = 0; i < vec.size(); i++) {
        vec[i] = distribution(gen);
    }
}

void InitVector(std::vector<float> &vec, float val) {
    for (int i = 0; i < vec.size(); i++) {
        vec[i] = val;
    }
}

int worker(cublasHandle_t handle, int M, int N, int K, int test_num_repeats) {
    const int lda = M;
    const int ldb = K;
    const int ldc = M;

    // Allocate device memory for matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    // Initialize matrices A and B on the host
    std::vector<float> h_A(M * K), h_B(K * N), C_ref(M * N, 0.f);
    InitVector(h_A);
    InitVector(h_B);

    // Copy matrices A and B to device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<double> elapsed_times;
    for (int i = 0; i < test_num_repeats; i++) {
        // Create cuBLAS events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Record the start event
        cudaEventRecord(start);

        // Perform matrix multiplication: C = alpha * A * B + beta * C
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, ldb, d_A, lda, &beta, d_C, ldc);

        // Record the stop event
        cudaEventRecord(stop);

        // Synchronize to make sure recording is finished
        cudaEventSynchronize(stop);

        // Calculate elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        elapsed_times.push_back(milliseconds);

        // Destroy cuBLAS events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Copy the result matrix C back to host
    std::vector<float> h_C(M * N);
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare with Ref
    float max_diff = 0.f;
    if (M < 1024) {
        RefGemm(M, N, K, h_A.data(), h_B.data(), C_ref.data());
        max_diff = Compare(M, N, h_C.data(), C_ref.data());
    }

    double mean = GetMeanTimeMillisecond(elapsed_times),
           median = GetMedianTimeMillisecond(elapsed_times),
           minimum = GetMinTimeMillisecond(elapsed_times);

    double gflops = (2.0 * M * N * K * 1.0e-6) / mean;

    // std::cout << "Matrix multiplication took " << milliseconds << " milliseconds." << std::endl;
    printf("Kernel: cuBlas, M=%d, N=%d, K=%d, mean=%.2fms, median=%.2fms, min=%.2fms, gflops=%f, max_diff=%f\n",
            M, N, K, mean, median, minimum, gflops, max_diff);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 1;
}

int main() {
    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Enable profiling
    cublasStatus_t status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to set pointer mode: " << status << std::endl;
        return 1;
    }

    const int test_num_repeats = TEST_NUM_REPEATS;
    for (int scale = TEST_SCALE_START; scale <= TEST_SCALE_END; scale *= TEST_SCALE_FACTOR) {
        const int M = OCL_GEMM_M < 0 ? scale : OCL_GEMM_M,
                  N = OCL_GEMM_N < 0 ? scale : OCL_GEMM_N,
                  K = OCL_GEMM_K < 0 ? scale : OCL_GEMM_K;
        worker(handle, M, N, K, test_num_repeats);
    }

    // Destroy cuBLAS shandle
    cublasDestroy(handle);

    return 0;
}
