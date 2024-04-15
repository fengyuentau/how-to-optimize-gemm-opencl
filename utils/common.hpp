#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

double GetMeanTimeMillisecond(std::vector<double> elapsed_times);
double GetMedianTimeMillisecond(std::vector<double> elapsed_times);
double GetMinTimeMillisecond(std::vector<double> elapsed_times);

void InitVector(std::vector<float> &vec);
void InitVector(std::vector<float> &vec, float val);

void RefGemm(const int M, const int N, const int K,
             const float *A,
             const float *B,
             float *C);
float Compare(const int M, const int N, const float *C, const float *C_ref);
