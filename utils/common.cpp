#include "common.hpp"

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
