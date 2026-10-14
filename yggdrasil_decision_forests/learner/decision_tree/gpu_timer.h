#ifndef YGG_LEARNER_DECISION_TREE_GPU_TIMER_H_
#define YGG_LEARNER_DECISION_TREE_GPU_TIMER_H_

#include <chrono>

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

inline TimePoint now() { return Clock::now(); }

inline double ms_since(TimePoint t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

inline double ms_between(TimePoint a, TimePoint b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

#endif  // YGG_LEARNER_DECISION_TREE_GPU_TIMER_H_
