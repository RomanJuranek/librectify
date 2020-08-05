#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "librectify.h"
#include "geometry.h"
#include "estimator.h"
#include "prosac.h"
#include "utils.h"
#include "line_pencil.h"

using namespace std;
using namespace Eigen;


namespace librectify {


LinePencilModel::LinePencilModel(const vector<LineSegment> & lines)
{
    h = homogeneous(lines).rowwise().normalized(); // homogenous coords of lines
    anchor = anchor_point(lines); // center point of lines segment
    direction = direction_vector(lines); // direction of line segment
    length = direction.rowwise().norm();
    direction.rowwise().normalize();
}


int LinePencilModel::size() const
{
    return h.rows();
}


int LinePencilModel::minimum_set_size() const
{
    return 2;
}


ArrayXf LinePencilModel::get_weights(const ArrayXi & indices) const
{
    float k = floor(ht_space_size/2.f);
    float k1 = k - 1;
    ArrayXXf accumulator = ArrayXXf::Zero(ht_space_size,ht_space_size);

    auto rng = std::mt19937();
    auto rand_idx = std::uniform_int_distribution<int>(0, indices.size()-1);

    for (int i = 0; i < ht_num_hypotheses; ++i)
    {
        int a = indices(rand_idx(rng));
        int b = indices(rand_idx(rng));
        Vector3f x = h.row(a).cross(h.row(b));
        if ((x.array().abs() < 0.0001f).all())  // Check invalid point
            continue;
        x.normalize();
        if (x.z() < 0.f)
            x = -x;
        int u = k1*x(0) + k;
        int v = k1*x(1) + k;
        accumulator.block<3,3>(u-1,v-1) += length(a) + length(b);
    }

    // cout << accumulator << endl;

    int max_u, max_v;
    float max_val = accumulator.maxCoeff(&max_u, &max_v);

    Vector3f p((max_u-k)/k1 , (max_v-k)/k1, 0);
    if (p.norm() > 1.f)
    {
        p /= p.norm();
    }

    p.z() = sqrt(1.f - (pow(p.x(),2.f) + pow(p.y(),2.f)));

    return inclination(anchor(indices,all), direction(indices,all), p).array().pow(4);
}


bool LinePencilModel::sample_check(const ArrayXi & indices) const
{
    // If two lines lie close to each other in homogeneous space, their
    // intersection is likely to be noisy - a degenerate solution.
    // Using this, the model solver will prefer lines which are far
    // from each other.
    assert(indices.size()==minimum_set_size());
    auto valid = (h.row(indices(0)) - h.row(indices(1))).norm() > degeneracy_tol;
    return valid;
}


LinePencilModel::hypothesis_type LinePencilModel::fit(const ArrayXi & indices) const
{
    // A cross product of two lies - an intersection point
    assert(indices.size()==minimum_set_size());
    auto a = h.row(indices(0));
    auto b = h.row(indices(1));
    return a.cross(b);
}


LinePencilModel::hypothesis_type LinePencilModel::fit_optimal(const ArrayXi & indices) const
{
    Matrix3f cov;
    if (indices.size() == 0)
    {
        cov = h.adjoint() * length.asDiagonal() * h;
    }
    else
    {
        VectorXf w = length(indices);
        cov = h(indices,all).adjoint() * w.asDiagonal() * h(indices,all);
    }
    SelfAdjointEigenSolver<Matrix3f> eig(cov);
    Index k;
    eig.eigenvalues().minCoeff(&k);
    VectorXf h = eig.eigenvectors().col(k);
    return h;
}


ArrayXf LinePencilModel::error(const hypothesis_type & h, const ArrayXi & indices) const
{
    return -inclination(anchor(indices,all), direction(indices,all), h).array() + 1.0f;
}


void estimate_line_pencils(
    vector<LineSegment> & lines,
    bool use_prosac
    )
{
    auto bb = bounding_box(lines);
    auto p = bbox_center(bb);
    float scale = bbox_size(bb).maxCoeff();
    auto lines_norm = normalize_lines(lines, p, scale);

    // normalize lines
    LinePencilModel model(lines_norm);
    ArrayXi groups;

    if (use_prosac)
    {
        PROSAC_Estimator<LinePencilModel> estimator;
        estimator.eta = 0.01;
        model.ht_num_hypotheses = 50000;
        model.ht_space_size = 257;
        groups = estimate_multiple_structures(estimator, model, 6, 0.001f, 0.004f);
    }
    else
    {
        DirectEstimator<LinePencilModel> estimator;
        model.ht_num_hypotheses = 50000;
        model.ht_space_size = 257;
        estimator.inlier_threshold = 0.998;
        groups = estimate_multiple_structures(estimator, model, 6, 0.001f, 0.004f);
    }

    // groups contains an id which can be just put into the group_id
    // Set group_id to original lines - same order
    for (int i = 0; i < lines.size(); ++i)
    {
        lines[i].group_id = groups(i);
    }
}


} // namespace