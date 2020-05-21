#pragma once

#include <vector>
#include <array>
#include <list>

#include <Eigen/Core>

#include "image.h"

namespace librectify {

Eigen::ArrayXXf gauss_deriv_kernel(int size, float sigma, bool dir_x);

void maximum_filter(const Image & image, Image & out, int size);

void binary_dilate(const Mask & image, Mask & out);

void conv_2d(const Image & image, const Image & kernel, Image & out);

void find_peaks(const Image & image, int size, float min_value, Eigen::MatrixX2i & loc);

void flood_init_mask(Mask & mask);

int flood(const Image & image, const Eigen::Vector2i seed, float tolerance, Mask & mask, Eigen::MatrixX2i & px_loc, Eigen::VectorXf & px_val);

} // namespace