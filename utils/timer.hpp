#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>

class TickMeter {
public:
    void Start() {
        start = std::chrono::high_resolution_clock::now();
        started = true;
    }

    void End() {
        end = std::chrono::high_resolution_clock::now();
        if (started) {
            elapsed_times.push_back(GetTimeMillisecond());
        }
        started = false;
    }

    void Reset() {
        elapsed_times.clear();
        started = false;
    }

    double GetTimeMillisecond() {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        return elapsed.count();
    }

    double GetMeanTimeMillisecond() {
        if (!started) {
            return std::accumulate(elapsed_times.begin(), elapsed_times.end(), 0.f) / (double)elapsed_times.size();
        } else {
            return -1;
        }
    }

    double GetMedianTimeMillisecond() {
        if (!started) {
            std::sort(elapsed_times.begin(), elapsed_times.end());

            auto half_length = static_cast<size_t>(elapsed_times.size() / 2);
            return elapsed_times.size() % 2 == 0 ?
                        (elapsed_times[half_length - 1] + elapsed_times[half_length]) / 2.0f :
                        elapsed_times[half_length];
        } else {
            return -1;
        }
    }

    double GetMinTimeMillisecond() {
        if (!started) {
            std::sort(elapsed_times.begin(), elapsed_times.end());

            return elapsed_times.front();
        } else {
            return -1;
        }
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;

    bool started{false};
    std::vector<double> elapsed_times;
};
