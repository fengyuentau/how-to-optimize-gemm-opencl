#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>

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
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        return elapsed.count() / 1000.0 / 1000.0;
    }

    void GetPerformanceResults(double &mean, double &median, double &minimum) {
        if (started) {
            End();
        } else {
            // drop the largest one
            auto max_it = std::max_element(elapsed_times.begin(), elapsed_times.end());
            if (max_it != elapsed_times.end()) {
                elapsed_times.erase(max_it);
            }

            mean = std::accumulate(elapsed_times.begin(), elapsed_times.end(), 0.f) / (double)elapsed_times.size();

            std::sort(elapsed_times.begin(), elapsed_times.end());

            auto half_length = static_cast<size_t>(elapsed_times.size() / 2);
            median = elapsed_times.size() % 2 == 0 ?
                        (elapsed_times[half_length - 1] + elapsed_times[half_length]) / 2.0f :
                        elapsed_times[half_length];
            minimum = elapsed_times.front();
        }
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;

    bool started{false};
    std::vector<double> elapsed_times;
};
