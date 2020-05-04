#include "liblgroup.h"

#include <Eigen/Core>
#include <omp.h>


namespace librectify {


#ifdef _OPENMP
static int num_threads = omp_get_max_threads();
#endif


void set_num_threads(int t)
{
    #ifdef _OPENMP
    num_threads = t;
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

} // namespace