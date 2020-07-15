#pragma once

#include <random>
#include <Eigen/Core>


namespace librectify {

void choice_knuth(int N, int n, std::mt19937 & rng, Eigen::ArrayXi & dst);

float binom(int n, int k);

template <typename T>
T clip(const T & a, const T & l, const T & h)
{
    return std::max(std::min(a, h), l);
}

} // namespace
