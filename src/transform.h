#pragma once

#include <vector>
#include <map>
#include <Eigen/Core>

#include "config.h"
#include "librectify.h"


namespace librectify {


using CornerTransform = Eigen::Matrix<float,4,3>;

Eigen::Vector3f fit_single_vanishing_points(const std::vector<LineSegment> & lines, int g);
std::map<int,Eigen::Vector3f> fit_vanishing_points(const std::vector<LineSegment> & lines);

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