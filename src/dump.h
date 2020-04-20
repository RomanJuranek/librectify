#pragma once

#include <Eigen/Core>
#include <fstream>
#include <sstream>

template <typename Derived>
void dump_array(const Eigen::Array<Derived,-1,-1, Eigen::RowMajor> & src,  std::string filename)
{
    std::ofstream f(filename);
    f << src;
}

template<class ArrayType>
std::string image_size(const ArrayType & arr)
{
    std::stringstream s;
    s << "(" << arr.rows() << "x" << arr.cols() << ")";
    return s.str();
}
