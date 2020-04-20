#pragma once

#include <vector>
#include <array>
#include <list>

#include <Eigen/Core>

#include "image.h"


typedef std::array<Eigen::Index,2> Location;

struct PeakPoint {
    int i, j;
    float v;
};

Eigen::ArrayXXf gauss_deriv_kernel(int size, float sigma, bool dir_x);

void maximum_filter(const Image & image, Image & out, int size);

void binary_dilate(const Mask & image, Mask & out);

void conv_2d(const Image & image, const Image & kernel, Image & out);

//void block_reduce(const Image & image, Image & out, int block_size);

//void box_filter_5x5(const Image & image, Image & out);

std::vector<PeakPoint> find_peaks(const Image & image, int size, float min_value);

void flood_init_mask(Mask & mask);

int flood(const Image & image, const Location seed, float tolerance, Mask & mask, Eigen::MatrixX2i & px_loc, Eigen::VectorXf & px_val);
