#include <iostream>
#include <vector>
#include <algorithm>

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
    return -(direction(indices,all) * Eigen::Vector2f(0,1)).cwiseAbs().array() + 1.0f;
    //return length(indices).eval();
    // VectorXf h = cascaded_hough_transform(h(indices,all), 32);
    // return inlier_score(h, indices);
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
    VectorXf w = length(indices);
    Matrix3f cov = h(indices,all).adjoint() * w.asDiagonal() * h(indices,all);
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


void estimate_line_pencils(vector<LineSegment> & lines)
{
    auto bb = bounding_box(lines);
    auto p = bbox_center(bb);
    float scale = bbox_size(bb).maxCoeff();
    
    auto lines_norm = normalize_lines(lines, p, scale);

    // normalize lines
    LinePencilModel model(lines_norm);
    // RANSAC_Estimator<LinePencilModel> estimator(10000);
    PROSAC_Estimator<LinePencilModel> estimator;
    //DirectEstimator<LinePencilModel> estimator;

    // run estimator on normalized lines
    ArrayXi groups = estimate_multiple_structures(estimator, model, 4, 0.002);

    // groups contains an id which can be just put into the group_id
    // Set group_id to original lines - same order
    for (int i = 0; i < lines.size(); ++i)
    {
        lines[i].group_id = groups(i);
    }
}


} // namespace