#include <random>
#include <Eigen/Core>

#include "math_utils.h"


using namespace std;
using namespace Eigen;


namespace librectify {


void choice_knuth
(
    int N,    // size of set sampling from
    int n,        // size of each sample
    std::mt19937 & rng,
    ArrayXi & dst  // output, zero-offset indicies to selected items
)
{
    int t = 0; // total input records dealt with
    int m = 0; // number of items selected so far
    double u;
    auto uniform = uniform_real_distribution<float>(0, 1);
    while (m < n)
    {
        u = uniform(rng); // call a uniform(0,1) random number generator
        if ( (N - t)*u >= n - m )
        {
            t++;
        }
        else
        {
            dst(m) = t;
            t++; m++;
        }
    }
}



float binom(int n, int k)
{  
    int res = 1;  
  
    // Since C(n, k) = C(n, n-k)  
    if ( k > n - k )  
        k = n - k;  
  
    // Calculate value of  
    // [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1]  
    for (int i = 0; i < k; ++i)  
    {  
        res *= (n - i);  
        res /= (i + 1);  
    }  
  
    return res;  
} 


} // namespace