/*

*/

#include <vector>
#include <queue>
#include <list>
#include <cmath>
#include <omp.h>

#include <Eigen/Core>

#include "librectify.h"
#include "dump.h"
#include "image.h"
#include "filter.h"
#include "threading.h"

#include <iostream>


using namespace Eigen;
using namespace std;


namespace librectify {


void maximum_filter(const Image & image, Image & out, int size)
{
    int n = 2*size+1;
    out.resizeLike(image);
    out.setZero();
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(get_num_threads()) if (is_omp_enabled())
    #endif
    for (int i = 0; i < image.rows()-n+1; ++i)
        for (int j = 0; j < image.cols()-n+1; ++j)
            out(i+size,j+size) = image.block(i,j,n,n).maxCoeff();
}


void binary_dilate(const Mask & image, Mask & out)
{
    int n = 3;
    out.resizeLike(image);
    out.setZero();
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(get_num_threads()) if (is_omp_enabled())
    #endif
    for (int i = 0; i < image.rows()-n+1; ++i)
        for (int j = 0; j < image.cols()-n+1; ++j)
        {
            out(i+1,j+1) = image.block<3,3>(i,j).maxCoeff() != 0;
        }
}


ArrayXXf gauss_deriv_kernel(int size, float sigma, bool dir_x)
{
    int n = 2*size+1;
    ArrayXXf H(n,n);
    for (int i = 0; i < H.rows(); ++i)
        for (int j = 0; j < H.cols(); ++j)
        {
            float x = float(j - size);
            float y = float(i - size);
            float z = float((dir_x) ? x : y);
            H(i,j) = z / float(2*M_PI*pow(sigma,4.0f)) * exp(-(pow(x,2.0f)+pow(y,2.0f))/(2*pow(sigma,2.0f)));
        }
    return H;
}


void conv_2d(const Image & image, const Image & kernel, Image & out)
{
    auto nr = kernel.rows();
    auto nc = kernel.cols();

    out.resizeLike(image);
    out.setZero();

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(get_num_threads()) if (is_omp_enabled())
    #endif
    for (int i = 0; i < image.rows()-nr+1; ++i)
        for (int j = 0; j < image.cols()-nc+1; ++j)
            out(i+nr/2,j+nc/2) = (image.block(i,j,nr,nc) * kernel).sum();
}


void flood_init_mask(Mask & mask)
{
    mask.row(0).setConstant(1);
    mask.row(mask.rows()-1).setConstant(1);
    mask.col(0).setConstant(1);
    mask.col(mask.cols()-1).setConstant(1);
}


int flood(const Image & image, const Vector2i seed, float tolerance, Mask & mask, MatrixX2i & px_loc, VectorXf & px_val)
{
    float seed_val = image(seed[0], seed[1]);
    float min_val = (1-tolerance) * seed_val;
    
    vector<Vector2i,Eigen::aligned_allocator<Vector2i> > res;
    res.reserve(256);

    queue<Vector2i> pixel_queue;
    pixel_queue.push(seed);
    while (!pixel_queue.empty())
    {
        auto p = pixel_queue.front();
        pixel_queue.pop();
        auto r = p.x();
        auto c = p.y();
        if (!mask(r,c) && (image(r,c) > min_val))
        {
            res.push_back(p);
            mask(r,c) = 1;
            pixel_queue.push({r,c-1});
            pixel_queue.push({r,c+1});
            pixel_queue.push({r+1,c});
            pixel_queue.push({r-1,c});
            pixel_queue.push({r-1,c-1});
            pixel_queue.push({r-1,c+1});
            pixel_queue.push({r+1,c-1});
            pixel_queue.push({r+1,c+1});
        }
    }

    auto n_pixels = res.size();
    px_loc.resize(n_pixels, 2);
    px_val.resize(n_pixels);
    int i = 0;
    for (auto & r: res)
    {
        px_loc.row(i) = r;
        px_val(i) = image(r.x(),r.y());
        ++i;
    }

    return int(n_pixels);
}


struct PeakPoint {
    int i, j;
    float v;
};

void find_peaks(const Image & image, int size, float min_value, MatrixX2i & loc)
{
    Image max_im;
    maximum_filter(image, max_im, size);
    auto peaks = ((max_im == image) && (image > min_value)).eval();

    vector<PeakPoint> res;
    res.reserve(1024);

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(get_num_threads()) if (is_omp_enabled())
    #endif
    for (int i = 0; i < peaks.rows(); ++i)
        for (int j = 0; j < peaks.cols(); ++j)
        {
            if (peaks(i,j))
            {
                #ifdef _OPENMP
                #pragma omp critical
                #endif
                res.push_back({i,j,image(i,j)});
            }
        }

    std::sort(res.begin(), res.end(), [](const PeakPoint & i, const PeakPoint & j) {return i.v > j.v;} );

    loc.resize(res.size(), 2);
    for (Index i = 0; i < res.size(); ++i)
    {
        loc.row(i) = Vector2i(res[i].i, res[i].j);
    }
}

}