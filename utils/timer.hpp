#include <chrono>

class TickMeter {
public:
    void Start() {
        start = std::chrono::high_resolution_clock::now();
    }

    void End() {
        end = std::chrono::high_resolution_clock::now();
    }

    double GetTimeMillisecond() {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        return elapsed.count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
};
