#pragma once

#include <vector>
#include <Eigen/Core>
#include "threading.h"
#include "librectify.h"


namespace librectify {


class LinePencilModel
{
    Eigen::MatrixX3f h;
    Eigen::MatrixX2f anchor;
    Eigen::MatrixX2f direction;
    Eigen::VectorXf length;
public:
    using hypothesis_type = Eigen::Vector3f;

    float degeneracy_tol {0.05};
    int ht_space_size {65};
    int ht_num_hypotheses {20000};

    LinePencilModel(const std::vector<LineSegment> & lines);
    int size() const;
    int minimum_set_size() const;
    Eigen::ArrayXf get_weights(const Eigen::ArrayXi & indices) const;
    bool sample_check(const Eigen::ArrayXi & indices) const;
    hypothesis_type fit(const Eigen::ArrayXi & indices) const;
    hypothesis_type fit_optimal(const Eigen::ArrayXi & indices) const;
    Eigen::ArrayXf error(const hypothesis_type & h, const Eigen::ArrayXi & indices) const;
    float inlier_score(const hypothesis_type & h, float tol, const Eigen::ArrayXi & indices) const;
};

void estimate_line_pencils(std::vector<LineSegment> & lines, int, float, float, const ThreadContext & ctx);

} // namespace