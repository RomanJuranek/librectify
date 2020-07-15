#pragma once

#include <vector>
#include <Eigen/Core>

#include "config.h"
#include "librectify.h"


namespace librectify {


// struct VanishingPoint
// {
//     Eigen::Vector3f coords; // in image space
//     float residual;  // Eigenvalue associated with the point - low means better
//     int n_inliers; // Support set size - sum of lengths
//     float support;

//     VanishingPoint(const Eigen::Vector3f & x):
//         coords(x), residual(0), n_inliers(0), support(0) {}
//     VanishingPoint(const Eigen::MatrixX3f & h, const Eigen::VectorXf & w);

//     float distance(const Eigen::Vector3f & c) const
//     {
//         if (coords.z() < EPS)
//             return INFINITY;
//         else
//             return Eigen::Vector2f(coords.x() - c.x(), coords.y() - c.y()).norm();
//     }

//     // Direction from center to the point
//     Eigen::Vector2f direction(const Eigen::Vector3f & c) const
//     {
//         return Eigen::Vector2f(coords.x() - c.x() * coords.z(), coords.y() - c.y() * coords.z()).normalized();
//     }

//     float quality() const
//     {
//         return support;
//     }
// };

using CornerTransform = Eigen::Matrix<float,4,3>;

Eigen::MatrixX3f fit_vanishing_points(const std::vector<LineSegment> & lines);

Eigen::Vector3f select_horizontal_point(
    const Eigen::MatrixX3f & vps,
    const Eigen::Vector3f & center,
    const Eigen::Vector3f & vertical,
    float min_distance);

Eigen::Vector3f select_vertical_point(
    const Eigen::MatrixX3f & vps,
    const Eigen::Vector3f & center,
    float angular_tolerance,
    float min_distance);

CornerTransform compute_image_transform(
    int width, int height,
    const Eigen::Vector3f & horizontal_vp,
    const Eigen::Vector3f & vertical_vp);


} // namespace