#include "liblgroup.h"
#include "threading.h"

#include <algorithm>
#include <omp.h>
#include <Eigen/Core>

namespace librectify {


#ifdef _OPENMP
static int num_threads = 0;
#endif


void set_num_threads(int t)
{
    #ifdef _OPENMP
    num_threads = std::min(t, omp_get_max_threads());
    Eigen::setNbThreads(t);
    #endif
}


int get_num_threads()
{
    int _t;
    #ifdef _OPENMP
    _t = num_threads;
    #else
    _t = 0;
    #endif
    return _t;
}


bool is_omp_enabled()
{
    return num_threads >= 0;
}

} // namespace