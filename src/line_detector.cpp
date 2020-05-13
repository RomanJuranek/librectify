/*
Image line segment detection

(c) 2019, Roman Juranek <ijuranek@fit.vutbr.cz>

Development of this software was funded by
TACR project TH04010394, Progressive Image Processing Algorithms.
*/

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <queue>
#include <set>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "line_detector.h"
#include "config.h"
#include "geometry.h"
#include "filter.h"
#include "utils.h"
#include "dump.h"
#include "threading.h"


using namespace Eigen;
using namespace std;

namespace librectify {


void image_gradients(const Image & image, Image & dx, Image & dy, Image & mag)
{
    auto Hx = gauss_deriv_kernel(EDGE_KERNEL_SIZE,EDGE_KERNEL_SIGMA,true);
    auto Hy = gauss_deriv_kernel(EDGE_KERNEL_SIZE,EDGE_KERNEL_SIGMA,false);
    conv_2d(image, Hx, dx);
    conv_2d(image, Hy, dy);
    mag = (dx.pow(2) + dy.pow(2)).sqrt();
}


void directional_gradient(const Image & dx, const Image & dy, float theta, Image & out)
{
    float st = sin(theta);
    float ct = cos(theta);
    out = (dx*st + dy*ct).abs().eval();
}


struct Component {
    MatrixX2i px_loc;
    VectorXf px_val;
};


vector<LineSegment> fit_lines_to_components(list<Component> & components)
{
    //cout << "Fitting line paramteres" << endl;
    vector<LineSegment> res;
    res.reserve(components.size());
    list<Component>::iterator c;
    #ifdef _OPENMP
    #pragma omp parallel private(c), num_threads(get_num_threads()) if (is_omp_enabled())
    #endif
    for (c=components.begin(); c != components.end(); ++c)
    {
        #ifdef _OPENMP
        #pragma omp single nowait
        #endif
        {
            LineSegment l = fit_line_parameters(c->px_loc.cast<float>(), c->px_val);
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            res.emplace_back(l);
        }
    }
    return res;
}


list<Component> find_components(
    vector<Image> & grad,
    vector<PeakPoint> & seed,
    vector<int> & seed_bin,
    float tolerance)
{
    Mask visited = Mask::Zero(grad[0].rows(), grad[0].cols());
    flood_init_mask(visited);
   
    list<Component> components;
   
    size_t i = 0;
    for (auto & s:seed)
    {
        bool was_visited = visited(s.i,s.j);
        size_t s_bin = seed_bin[i];
        if (!was_visited)
        {
            Component c;
            int size = flood(grad[s_bin], {s.i, s.j}, tolerance, visited, c.px_loc, c.px_val);
            if (size > COMPONENT_MIN_SIZE)
            {
                //#if LGROUP_DEBUG_PRINTS
                //cerr << "Adding component of size " << c.px_val.size() << ", seed point " << s.i << "," << s.j << endl;
                //#endif
                components.emplace_back(c);
            }
        }
        ++i;
    } // seed loop

    return components;
}

using timept = chrono::steady_clock::time_point;

void gradient_directions(const Image & dx, const Image & dy, int n_bins, Image_int & grad_bin, vector<Image> & grad)
{
    grad_bin.resizeLike(dx);
    Image grad_max;
    grad_max.resizeLike(dx);
    grad_max.setZero();
    grad.clear();
    grad.resize(n_bins);

    #if LGROUP_DEBUG_PRINTS
    timept t0 = std::chrono::steady_clock::now();
    #endif

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(get_num_threads()) if (is_omp_enabled())
    #endif
    for (int i = 0; i < n_bins; i++)
    {
        float theta = float(i*M_PI) / n_bins;
        grad[i] = (dx*sin(theta) + dy*cos(theta)).abs();
    }

    #if LGROUP_DEBUG_PRINTS
    timept t1 = std::chrono::steady_clock::now();
    #endif

    for (int i = 0; i < n_bins; i++)
    {
        grad_bin = (grad[i] > grad_max).select(i, grad_bin);
        grad_max = grad_max.max(grad[i]);
    }

    #if LGROUP_DEBUG_PRINTS
    timept t2 = std::chrono::steady_clock::now();
    #endif

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(get_num_threads()) if (is_omp_enabled())
    #endif
    for (int i = 0; i < n_bins; i++)
    {
        Mask mask;
        binary_dilate((grad_bin==i), mask);
        //grad[i] *= mask.cast<float>();
        grad[i] = mask.select(grad[i],0);
    }

    #if LGROUP_DEBUG_PRINTS
    timept t3 = std::chrono::steady_clock::now();
    #endif
    
    #if LGROUP_DEBUG_PRINTS
    clog << "gradient_directions: energy: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " [ms]" << endl;
    clog << "gradient_directions: max bin: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " [ms]" << endl;
    clog << "gradient_directions: masks: " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " [ms]" << endl;
    #endif
}

vector<LineSegment> postprocess_lines_segments(const vector<LineSegment> & lines);
vector<LineSegment> find_line_segments(const Image & image, int seed_dist, float seed_ratio, float mag_tolerance)
{
    #if LGROUP_DEBUG_PRINTS
    clog << "find_line_segments: image size " << image_size(image) << endl;
    timept t0 = std::chrono::steady_clock::now();
    #endif

    Image dx, dy, mag;
    image_gradients(image, dx, dy, mag);

    #if LGROUP_DEBUG_PRINTS
    timept t1 = std::chrono::steady_clock::now();
    #endif

    // Gradient bin for each pixel - orientation with highest magnitude
    Image_int grad_bin;
    // Vector of gradient images
    vector<Image> grad;
    gradient_directions(dx, dy, 8, grad_bin, grad);

    #if LGROUP_DEBUG_PRINTS
    timept t2 = std::chrono::steady_clock::now();
    #endif

    float min_seed_value = mag.maxCoeff() * (1 - max(min(seed_ratio,1.f), 0.f));
    vector<PeakPoint> seed = find_peaks(mag, seed_dist, min_seed_value);
    vector<int> seed_bin(seed.size());
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(get_num_threads()) if (is_omp_enabled())
    #endif
    for (int i = 0; i < int(seed.size()); ++i)
    {
        seed_bin[i] = grad_bin(seed[i].i, seed[i].j);
    }

    #if LGROUP_DEBUG_PRINTS
    clog << "find_line_segments: " << seed.size() << " seed points" << endl;
    timept t3 = std::chrono::steady_clock::now();
    #endif


    auto components = find_components(grad, seed, seed_bin, mag_tolerance);
    #if LGROUP_DEBUG_PRINTS
    timept t4 = std::chrono::steady_clock::now();
    #endif


    auto lines = fit_lines_to_components(components);
    #if LGROUP_DEBUG_PRINTS
    clog << "find_line_segments: " << lines.size() << " line segments" << endl;
    timept t5 = std::chrono::steady_clock::now();
    #endif


    #if LGROUP_DEBUG_PRINTS
    clog << "Total line detection: " << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t0).count() << " [ms]" << endl;
    clog << "gradients: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " [ms]" << endl;
    clog << "Direction energy: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " [ms]" << endl;
    clog << "Seed points: " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " [ms]" << endl;
    clog << "Components: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << " [ms]" << endl;
    clog << "Line fitting: " << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count() << " [ms]" << endl;
    #endif

    return lines;
}


LineSegment merge_lines(const vector<LineSegment> & lines)
{
    if (lines.size() == 1)
        return lines[0];
    size_t n = 2*lines.size();
    MatrixX2f X(n, 2);
    VectorXf W(n);
    VectorXf l = length(lines);
    auto wts = l.cwiseProduct(weigths(lines)).eval();
    for (size_t i = 0; i < lines.size(); ++i)
    {
        const auto & ln = lines[i]; 
        X.row(2*i+0) = Vector2f(ln.y1, ln.x1);
        X.row(2*i+1) = Vector2f(ln.y2, ln.x2);
        W(2*i+0) = wts(i);
        W(2*i+1) = wts(i);
    }
    LineSegment merged = fit_line_parameters(X, W);
    merged.weight = wts.sum() / l.sum();
    return merged;
}


void dfs(int v, const SparseMatrix<float> & A, Array<bool,-1,1> & visited, ArrayXi & components, int label)
{
    queue<int> nodes;
    nodes.push(v);
    while (!nodes.empty())
    {
        int n = nodes.front();
        nodes.pop();
        visited(n) = true;
        components(n) = label;
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(get_num_threads()) if (is_omp_enabled())
        #endif
        for (Index u = n+1; u < A.cols(); ++u)
        {
            if (A.coeff(n,u) > 0)
            {
                if (!visited(u))
                {
                    #ifdef _OPENMP
                    #pragma omp critical
                    #endif
                    {
                        nodes.push(int(u));
                    }
                }
            }
        }
    }
}

ArrayXi graph_components(const SparseMatrix<float> & A)
{
    Index nv = A.rows();
    Array<bool,-1,1> visited(nv);
    visited.setConstant(false);
    ArrayXi components(nv);
    for (Index v = 0; v < nv; ++v)  // vertices
    {
        if (!visited(v))
        {
            //cout << "Component " << v << endl;
            dfs(int(v), A, visited, components, int(v));
        }
    }
    return components;
}


vector<LineSegment> postprocess_lines_segments(const vector<LineSegment> & lines)
{
    size_t n_lines = lines.size();
    //timept t0 = std::chrono::steady_clock::now();

    // Get pairwise distances
    MatrixX2f d = direction_vector(lines).rowwise().normalized();
    VectorXf l = length(lines);
    MatrixX2f n(n_lines, 2);
    n << -d.col(1), d.col(0);

    // cout << "d=\n" << d <<endl;
    // cout << "n=\n" << n <<endl;

    SparseMatrix<float> aff(n_lines,n_lines);

    Matrix2f A, B, U, V;
    #ifdef _OPENMP
    #pragma omp parallel for private(A,B,U,V) schedule(dynamic,1), num_threads(get_num_threads()) if (is_omp_enabled())
    #endif
    for (Index i = 0; i < Index(n_lines); ++i)
    {
        const LineSegment & li = lines[i];
        A << li.x1, li.y1, li.x2, li.y2;
        U.col(0) = d.row(i);
        U.col(1) = n.row(i);

        for (Index j = i+1; j < Index(n_lines); ++j)
        {
            const LineSegment & lj = lines[j];
            B << lj.x1, lj.y1, lj.x2, lj.y2;
            V.col(0) = d.row(j);
            V.col(1) = n.row(j);
            if ( abs((U.adjoint() * V)(0,0)) < 0.95)  // cos(max_angular_difference)
            {
                continue;
            }
            Matrix2f W;
            if (l(i) < l(j))
            {
                W.noalias() = ((A.rowwise() - B.row(0)) * V) / l(j);
            }
            else
            {
                W.noalias() = ((B.rowwise() - A.row(0)) * U) / l(i);
            }

            if (W.col(1).cwiseAbs().maxCoeff() < 0.05)
            {
                auto x = W.col(0).array();
                if ((x > -0.2).any() && (x < 1.2).any())
                {
                    //cerr << l(i) << "," << l(j) << endl;
                    //cerr << "A=\n" << A << endl;
                    //cerr << "B=\n" << B << endl;
                    // cerr << "V=\n" << V << endl;
                    // cerr << "U=\n" << U << endl;
                    // cerr << "W=\n" << W << endl;
                    #ifdef _OPENMP
                    #pragma omp critical
                    #endif
                    aff.insert(i,j) = 1;
                }
            }
        }
    }

    //cout << d << endl;

    //cout << aff.nonZeros() << endl;
    //cout << aff.rows() << "," << aff.cols() << endl;

    //timept t1 = std::chrono::steady_clock::now();

    // Get components and unique labels
    ArrayXi components = graph_components(aff);
    //cerr << components << endl;
    std::set<int> labels(components.begin(), components.end());
    //timept t2 = std::chrono::steady_clock::now();
    //cerr << components.size() << "," << lines.size() << endl;
    vector<LineSegment> res;
    res.reserve(labels.size());
    // For each unique label
    for (auto lbl : labels)
    {
        // Get indices of lines with the same label
        //ArrayXi idx = index_array(components == *lbl);
        //cerr << "set: " << *lbl << ", " << idx.size() << "," << (components == *lbl).count() << endl;
        //cerr << "l: " << lbl << endl;
        // Make copy to temporary vector
        vector<LineSegment> lbl_lines;
        lbl_lines.reserve(4);
        for (Index j = 0; j < components.size(); ++j)
            // copy idx[j]-th line to j-th position
            //lbl_lines[j] = lines[idx(j)];
            if (components(j) == lbl)
                lbl_lines.push_back(lines[size_t(j)]);
        // So now lbl_lines is a set of lines to be merged
        LineSegment merged = merge_lines(lbl_lines);
        res.push_back(merged);
    }
    //timept t3 = std::chrono::steady_clock::now();

    // clog << "distance: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " [ms]" << endl;
    // clog << "components: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " [ms]" << endl;
    // clog << "merging: " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " [ms]" << endl;
    // cout << "Original lines: " << lines.size() << ". Merged:" << res.size() << endl;
    // cerr << res.size() << endl;
    return res;
}

}