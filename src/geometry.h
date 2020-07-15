#pragma once

#include <vector>
#include <Eigen/Core>
#include "librectify.h"


namespace librectify {

// Line segment estimation

LineSegment fit_line_parameters(const Eigen::MatrixX2f & X, const Eigen::VectorXf & w);

// Line segment

Eigen::Vector3f homogeneous(const LineSegment & l);
Eigen::Vector2f anchor_point(const LineSegment & l);
Eigen::Vector2f direction_vector(const LineSegment & l);
Eigen::Vector2f normal_vector(const LineSegment & l);
float length(const LineSegment & l);

// Line segment collections

Eigen::MatrixX3f homogeneous(const std::vector<LineSegment> & lines);
Eigen::MatrixX2f anchor_point(const std::vector<LineSegment> & lines);
Eigen::MatrixX2f direction_vector(const std::vector<LineSegment> & lines);
Eigen::VectorXf reprojection_error(const std::vector<LineSegment> & lines);
Eigen::MatrixX2f normal_vector(const std::vector<LineSegment> & lines);
Eigen::VectorXf length(const std::vector<LineSegment> & lines);
Eigen::VectorXf weights(const std::vector<LineSegment> & lines);
Eigen::ArrayXi group_id(const std::vector<LineSegment> & lines);
std::vector<LineSegment> normalize_lines(const std::vector<LineSegment> & lines, const Eigen::Vector2f & p, float s);

// Bounding box

Eigen::Vector4f bounding_box(const std::vector<LineSegment> & lines);
Eigen::Vector2f bbox_center(const Eigen::Vector4f & bb);
Eigen::Vector2f bbox_size(const Eigen::Vector4f & bb);

// Points and lines in homogeneous space

Eigen::VectorXf inclination(const Eigen::MatrixX2f & a, const Eigen::MatrixX2f & d, const Eigen::RowVector3f & p);
Eigen::Vector3f normalize_point(const Eigen::Vector3f & p);
Eigen::Vector2f direction(const Eigen::Vector3f & a, const Eigen::Vector3f & b);
float distance(const Eigen::Vector3f & a, const Eigen::Vector3f & b);

}