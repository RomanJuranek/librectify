#pragma once

#include <vector>
#include <Eigen/Core>
#include "liblgroup.h"


LineSegment fit_line_parameters(const Eigen::MatrixX2f & X, const Eigen::VectorXf & w);

Eigen::Vector3f homogeneous(const LineSegment & l);
Eigen::Vector2f anchor_point(const LineSegment & l);
Eigen::Vector2f direction_vector(const LineSegment & l);
Eigen::Vector2f normal_vector(const LineSegment & l);
float length(const LineSegment & l);

Eigen::Vector4f bounding_box(const std::vector<LineSegment> & lines);
Eigen::MatrixX3f homogeneous(const std::vector<LineSegment> & lines);
Eigen::MatrixX2f anchor_point(const std::vector<LineSegment> & lines);
Eigen::MatrixX2f direction_vector(const std::vector<LineSegment> & lines);
Eigen::VectorXf reprojection_error(const std::vector<LineSegment> & lines);
Eigen::MatrixX2f normal_vector(const std::vector<LineSegment> & lines);
Eigen::VectorXf length(const std::vector<LineSegment> & lines);
Eigen::VectorXf weigths(const std::vector<LineSegment> & lines);
Eigen::ArrayXi group_id(const std::vector<LineSegment> & lines);

Eigen::VectorXf inclination(const Eigen::MatrixX2f & a, const Eigen::MatrixX2f & d, const Eigen::RowVector3f & p);