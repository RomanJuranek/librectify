#include <iostream>
#include <vector>
#include <list>
#include <numeric>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "config.h"
#include "librectify.h"
#include "geometry.h"
#include "threading.h"


using namespace std;
using namespace Eigen;

namespace librectify {


class RandomSampler
{
    std::mt19937 rng;
    int range;  // [0, range-1]
    unordered_set<int> res;
public:
    RandomSampler()
        :rng(random_device()()) {
            reset(0);
        }
    RandomSampler(const RandomSampler & other)
        :rng(random_device()())
        {
            reset(0);
        }
    void reset(int k)
    {
        range = k;
        res.clear();
    }
    int next()
    {
        if (res.size() == range)
            throw range_error("Exceeded range of random numbers");
        auto random_index = uniform_int_distribution<int>(0, int(range-1));
        std::pair<std::unordered_set<int>::iterator,bool> ret;
        do
        {
            ret = res.insert(random_index(rng));
        } while (!ret.second);
        return *ret.first;
    }
};



class LinePencilModel
{
    MatrixX3f h;
    MatrixX2f anchor;
    MatrixX2f direction;
    VectorXf length;
    VectorXf err;

public:
    LinePencilModel(const vector<LineSegment> & lines)
    {
        h = homogeneous(lines); // homogenous coords of lines
        anchor = anchor_point(lines); // center point of lines segment
        direction = direction_vector(lines); // direction of line segment
        length = direction.rowwise().norm();
        direction.rowwise().normalize();
        err = reprojection_error(lines);
    }

    int size() const
    {
        return int(h.rows());
    }

    int complexity() const
    {
        return 2;
    }

    Vector3f fit(vector<int> indices)
    {
        auto a = h.row(indices[0]);
        auto b = h.row(indices[1]);
        return a.cross(b);
    }

    Array<bool,-1,1> inliers(const VectorXf & p, const vector<int> & indices, float tolerance)
    {
        return (-inclination(anchor(indices,all), direction(indices,all), p).array() + 1.0f) < tolerance;
    }

    ArrayXf fittness(
        const VectorXf & p,
        const vector<int> & indices,
        float tolerance) const
    {
        ArrayXf angle_error = -inclination(anchor(indices,all), direction(indices,all), p).array() + 1.0f;
        ArrayXf length_score = length(indices).array();
        return (angle_error < tolerance).cast<float>() * length_score;
    }
};



// select m random values from first n elements
void sample(
    vector<int> & indices,
    int m,
    int n,
    vector<int>::iterator dst,
    RandomSampler & rng)
{
    rng.reset(n);
    for (int i = 0; i < m; ++i)
    {
        *dst++ = indices[rng.next()];
    }
}


VectorXf RANSAC(LinePencilModel & model, vector<int> indices, int max_iter, float tolerance)
{
    float best_fit = 0;
    VectorXf best_h;

    RandomSampler rng;
    #ifdef _OPENMP
    #pragma omp parallel for firstprivate(rng) shared(best_h, best_fit) num_threads(get_num_threads()) if (is_omp_enabled())
    #endif
    for (int iter=0; iter < max_iter; ++iter)
    {
        vector<int> samples(2);
        sample(indices, 2, int(indices.size()), samples.begin(), rng);
        VectorXf h = model.fit(samples);
        ArrayXf f = model.fittness(h, indices, tolerance);
        float fit = f.sum();
        // cerr << "h=" << h << ", fit=" << fit << endl;
        // abort();
        #ifdef _OPENMP
        #pragma omp critical (RANSAC)
        #endif
        if (fit > best_fit)
        {
            #if LGROUP_DEBUG_PRINTS
            clog << "RANSAC: iter=" << iter << ", fit=" << fit << endl;
            #endif
            best_h = h;
            best_fit = fit;
        }
    }
    //cerr << best_h << endl;
    return best_h;
}

/*
double binom(int n, int k)  
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


inline double P(int i, double beta, int n, int m)
{
    return std::pow(beta,(i - m)) * std::pow((1 - beta),(n - i + m)) * binom(n - m, i - m);
}


double Imin_value(int j, double beta, int n, int m)
{
    double I = 0;
    for (int i=j; i < n+1; i++)
        I += P(i, beta, n, m);
    return I;
}

*/

/*

// PROSAC based on https://willguimont.github.io/cs/2019/12/26/prosac-algorithm.html
VectorXf PROSAC(
    LinePencilModel & model,
    vector<int> & indices,
    vector<float> & quality,
    float tolerance,
    float beta,
    float phi,
    float eta)
{
    // TODO: sort indices acccording to quality
    using T = pair<int,float>;
    vector<T> q_idx(indices.size());
    for (size_t i = 0; i < q_idx.size(); ++i)
    {
        q_idx[i] = {indices[i], quality[i]};
    }

    sort(q_idx.begin(), q_idx.end(), [](const T & a, const T & b) {return a.second>b.second;} );

    vector<int> sorted_indices(indices.size());
    for (size_t i = 0; i < q_idx.size(); ++i)
    {
        sorted_indices[i] = q_idx[i].first;
        // cout << q_idx[i].second << ", ";
    }

    int N = model.size();
    int m = model.complexity();
    int t = 0; // Iteration counter
    int n = m; // Size of sampling set
    int n_star = N;
    int Tn = 1;
    int Tn_prime = 1;
    VectorXf best_h;
    int best_fit = 0;
    while (true)
    {
        t++;
        // Hypothesis generation set
        if (t == Tn_prime && n < n_star)
        {
            int Tn_1 = Tn * (n+1) / (n+1-m);
            Tn_prime += ceil(Tn_1 - Tn);
            Tn = Tn_1;
            n++;
        }
        //cout << "t=" << t << ", n=" << n << endl;
        // Sample from data
        // TODO: index n + [m-1] random samples
        vector<int> samples(m);
        if (n < N)
        {
            sample(sorted_indices, m-1, n-1, samples.begin());
            samples.back() = n;
        }
        else
        {
            sample(sorted_indices, m, n, samples.begin());
        }
        
        // Estimate parameters
        VectorXf h = model.fit(samples);

        // Verification
        VectorXf err = model.error(h, indices);
        int num_inliers = (err.array() < tolerance).count();
        //cout << num_inliers << endl;

        if (num_inliers > best_fit)
        {
            best_h = h;
            best_fit = num_inliers;
        }

        int imin = 0;
        for (int j = m; j < N-m; ++j)
        {
            if (Imin_value(j,beta,n,m) < phi)
            {
                imin = j;
                break;
            }
        }
        //cout << num_inliers << ", " << imin << endl;
        bool non_random = num_inliers > imin;

        double Pin = binom(num_inliers, m) / binom(n_star, m);
        bool maximality = pow(1 - Pin, t) <= eta;

        //cout << num_inliers << ", " << imin << ", " << Pin << endl;
        if (non_random && maximality)
            break;
    }
    //cout << "PROSAC finished after " << t << " iterations" << endl;
    return best_h;
}
*/


template<typename Derived, class OutputIterator>
void partition_indices(
    const vector<int> & indices,
    const ArrayBase<Derived> & predicate,
    OutputIterator true_inds,
    OutputIterator false_inds)
{
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        if (predicate(i))
        {
            *true_inds++ = idx;
        }
        else
        {
            *false_inds++ = idx;
        }
    }
}


template<class OutputIterator>
int estimate_single_line_group(
    LinePencilModel & model,
    vector<int> & indices,
    int group_id,
    float tolerance,
    OutputIterator inls_inds,
    OutputIterator inls_groups,
    OutputIterator outl_inds)
{
    VectorXf p = RANSAC(model, indices, RANSAC_MAX_ITER, tolerance);
    auto inliers = model.inliers(p, indices, tolerance);
    partition_indices(indices, inliers, inls_inds, outl_inds);
    int n_inliers = int(inliers.count());
    for (int i=0; i<n_inliers; ++i)
        *inls_groups++ = group_id;
    return n_inliers;
}


vector<LineSegment> group_lines(vector<LineSegment> & lines)
{
    LinePencilModel model(lines);

    vector<int> inl_indices;
    vector<int> inl_groups;
    vector<int> outl_indices(lines.size());
    iota(outl_indices.begin(), outl_indices.end(), 0);
    
    for (int g = 0; g < MAX_GROUPS; ++g)
    {
        vector<int> tmp_outl_indices;
        int inl_count = estimate_single_line_group(model, outl_indices, g, INLIER_TOLERANCE, back_inserter(inl_indices), back_inserter(inl_groups), back_inserter(tmp_outl_indices));
        outl_indices = tmp_outl_indices;
        #if LGROUP_DEBUG_PRINTS
        clog << "group_lines: group " << g << ": #outliers: " << tmp_outl_indices.size() << ", #inliers: " << inl_count << endl;
        #endif
        if (tmp_outl_indices.size() < MIN_LINES)
        {
            break;
        }
    }

    vector<LineSegment> res;
    res.reserve(lines.size());

    for (size_t i = 0; i < inl_indices.size(); ++i)
    {
        int idx = inl_indices[i];
        int group = inl_groups[i];
        auto l = lines[idx];
        l.group_id = group;
        res.push_back(l);
    }

    for (size_t i = 0; i < outl_indices.size(); ++i)
    {
        int idx = outl_indices[i];
        int group = -1;
        auto l = lines[idx];
        l.group_id = group;
        res.push_back(l);
    }

    assert(res.size() == lines.size());

    return res;
}


} // namespace