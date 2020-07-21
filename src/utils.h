#pragma once

#include <iterator>
#include <numeric>
#include <algorithm>

#include <Eigen/Core>


namespace librectify {


/*
Return the indices of elements that evaluate to true.
*/
template <typename Derived>
Eigen::ArrayXi nonzero(const Eigen::ArrayBase<Derived> & x)
{
    Eigen::Index n = x.count();
    Eigen::ArrayXi idx(n);
    int k = 0;
    for (Eigen::Index i = 0; i < x.size(); ++i)
    {
        if (x(i))
        {
            idx(k++) = int(i);
        }
    }
    return idx;
}


/*
Return the order of elements in descending order.
*/
template <typename Derived>
Eigen::ArrayXi argsort(const Eigen::DenseBase<Derived> & x)
{
    Eigen::ArrayXi idx(x.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
       [&x](size_t i, size_t j) {return x(i) > x(j);});
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