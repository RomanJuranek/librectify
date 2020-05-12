#pragma once


#include <iterator>
#include <algorithm>

#include <Eigen/Core>

namespace librectify {

template <typename Derived>
Eigen::ArrayXi index_array(const Eigen::ArrayBase<Derived> & x)
{
    Index n = x.count();
    Eigen::ArrayXi idx(n);
    int k = 0;
    for (Index i = 0; i < x.size(); ++i)
    {
        if (x(i))
        {
            idx(k++) = int(i);
        }
    }
    return idx;
}


template <class II, typename Derived, typename Func>
void transfrom_to_matrix(II first, II last, Func f, Derived & res)
{
    size_t n = std::distance(first, last);
    int cols = res.cols();
    res.resize(n, cols);
    std::transform(first, last, res.rowwise().begin(), f);
}

} // namespace