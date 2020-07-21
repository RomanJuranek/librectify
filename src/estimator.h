#pragma once

#include <vector>
#include <random>
#include <iostream>

#include <Eigen/Core>

#include "utils.h"
#include "math_utils.h"

namespace librectify
{


template <typename ModelType>
class Estimator
{
public:
    virtual typename ModelType::hypothesis_type solve(const ModelType & model, const Eigen::ArrayXi & indices, float tol) = 0;
};


template <typename ModelType>
class RANSAC_Estimator: Estimator<ModelType>
{
    int N;
    std::mt19937 rng;
public:
    using hypothesis_type = typename ModelType::hypothesis_type;
    RANSAC_Estimator(int max_iter) :N(max_iter), rng(std::random_device()())
    { }
    hypothesis_type solve(const ModelType & model, const Eigen::ArrayXi & indices, float tol)
    {
        hypothesis_type best_h;
        int I_N_best = 0;
        Eigen::ArrayXf best_err;
        
        Eigen::ArrayXi sample(model.minimum_set_size());
        Eigen::ArrayXi sample_indices(model.minimum_set_size());

        for (int i = 0; i < N; ++i)
        {
            choice_knuth(indices.size(), model.minimum_set_size(), rng, sample);
            sample_indices = indices(sample);

            if (!model.sample_check(sample_indices)) continue;
            auto h = model.fit(sample_indices);
            auto err = model.error(h, indices);
            int I_N = (err < tol).count();

            //std::cerr << sample_indices.transpose() << ", " << "I_N=" << I_N << ", h= " << h.transpose() << std::endl;

            if (I_N > I_N_best)
            {
                //std::cerr << i << ": " << I_N << std::endl;
                I_N_best = I_N;
                best_h = h;
                best_err = err;
                //std::cerr << h.transpose() << std::endl;
            }
        }
        //std::cerr << "---" <<std::endl;

        auto inliers = nonzero(best_err < tol);
        return model.fit_optimal(indices(inliers));
    }
};


template <typename ModelType>
class DirectEstimator: Estimator<ModelType>
{
public:
    using hypothesis_type = typename ModelType::hypothesis_type;
    float inlier_threshold {0.95};

    DirectEstimator() { }
    hypothesis_type solve(const ModelType & model, const Eigen::ArrayXi & indices, float tol)
    {
        Eigen::ArrayXf weights = model.get_weights(indices);
        auto inls = nonzero(weights > inlier_threshold);
        return model.fit_optimal(indices(inls));
    }
};


template <typename EstimatorType, typename ModelType>
Eigen::ArrayXi estimate_multiple_structures(
    EstimatorType & estimator,
    const ModelType & model,
    int max_structures,
    float tol)
{
    // ID of structure for each observation - inlier_flag[k] >= 0 means
    // that the observation k is inlier.
    Eigen::ArrayXi inlier_flag(model.size());
    inlier_flag.setConstant(-1);

    // Must be consistent with (inlier_flag < 0).sum()
    // Updated in each iteration
    int num_observations = model.size();

    // Estimating at most max_structures
    int k = 0;
    while (num_observations >= model.minimum_set_size() && k < max_structures)
    {
        // List of unassigned observations
        Eigen::ArrayXi observation_indices = nonzero(inlier_flag < 0);
        // Solve the model
        auto h = estimator.solve(model, observation_indices, tol);
        // Obtain inliers and flag them
        Eigen::ArrayXi inliers_k = observation_indices(nonzero(model.error(h, observation_indices)<2*tol));
        inlier_flag(inliers_k) = k;
        // Update number of unassigned observations
        num_observations -= inliers_k.size();
        ++k;
    }
    return inlier_flag;
}


}