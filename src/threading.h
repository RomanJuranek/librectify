#pragma once

#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace librectify {

class ThreadContext
{
private:
    int num_threads;
public:
    ThreadContext(int t)
    {
        set_num_threads(t);
    }
    int get_num_threads() const
    {
        return num_threads;
    }
    bool enabled() const
    {
        return num_threads >= 0;
    }
    void set_num_threads(int t)
    {
        num_threads = std::min(t, omp_get_max_threads());
    }
};

}