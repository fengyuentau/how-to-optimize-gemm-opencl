#include "init_vector.hpp"

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
