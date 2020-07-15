#pragma once

#include <cstdio>
#include <random>
#include <Eigen/Core>

#include "estimator.h"
#include "math_utils.h"


using std::cerr;
using std::endl;
using std::printf;


namespace librectify {


// Chi-square table for k=1 and p=(0,0.2)
// precalulated by scipy implementation chi2.isf( np.arange(0,0.2,0.01), 1 )
static const float chi2_table[20] = {
Eigen::Infinity, 6.6348966 , 5.41189443, 4.70929225, 4.21788459,
     3.84145882, 3.5373846 , 3.28302029, 3.06490172, 2.8743734,
     2.70554345, 2.55422131, 2.41732093, 2.29250453, 2.17795916,
     2.07225086, 1.97422609, 1.88294329, 1.79762406, 1.71761761};



/// Computation of the Maximum number of iterations for Ransac
/// with the formula from [HZ] Section: "How many samples" p.119
static inline
int niter_RANSAC(double p, // probability that at least one of the random samples picked up by RANSAC is free of outliers
                 double epsilon, // proportion of outliers
                 int s, // sample size
                 int Nmax) // upper bound on the number of iterations (-1 means INT_MAX)
{
    // compute safely N = ceil(log(1. - p) / log(1. - exp(log(1.-epsilon) * s)))
    double logarg, logval, N;
    if (Nmax == -1) {
        Nmax = INT_MAX;
    }
    assert(Nmax >= 1);
    if (epsilon <= 0.) {
        return 1;
    }
    // logarg = -(1-epsilon)^s
    logarg = -exp(s*log(1.-epsilon)); // C++/boost version: logarg = -std::pow(1.-epsilon, s);
    // logval = log1p(logarg)) = log(1-(1-epsilon)^s)
    logval = log(1.+logarg); // C++/boost version: logval = boost::math::log1p(logarg)
    N = log(1.-p) / logval;
    if (logval  < 0. && N < Nmax) {
        return (int)ceil(N);
    }
    return Nmax;
}


template <typename ModelType>
class PROSAC_Estimator: Estimator<ModelType>
{
    float eta {0.001};
    float beta {0.001};
    float psi {0.01};
    float p_good_sample {0.5};
    float max_outlier_proportion {0.8};
    std::mt19937 rng;

    float chi2_value;

    // * Non-randomness: eq. (7) states that i-m (where i is the cardinal of the set of inliers for a wrong
    // model) follows the binomial distribution B(n,beta). http://en.wikipedia.org/wiki/Binomial_distribution
    // For n big enough, B(n,beta) ~ N(mu,sigma^2) (central limit theorem),
    // with mu = n*beta and sigma = sqrt(n*beta*(1-beta)).
    // psi, the probability that In_star out of n_star data points are by chance inliers to an arbitrary
    // incorrect model, is set to 0.05 (5%, as in the original paper), and you must change the Chi2 value if
    // you chose a different value for psi.
    int Imin(int m, int n) {
        double mu = n*beta;
        double sigma = sqrt(n*beta*(1-beta));
        // Imin(n) (equation (8) can then be obtained with the Chi-squared test with P=2*psi=0.10 (Chi2=2.706)
        return (int)ceil(m + mu + sigma*sqrt(chi2_value));
    }

    /*
    This is workaround using tabulated chi2.isf(p,1). The correct solution
    infolves calculation of inverse regularized gamma function which may be added
    later. This should work reasonable.
    */
    float chi2(float p)
    {
        int i = floor(clip(p, 0.01f, 0.2f) * 100);
        return chi2_table[i];
    }

public:
    using hypothesis_type = typename ModelType::hypothesis_type;
    PROSAC_Estimator(): 
        rng(std::random_device()())
    {
        chi2_value = chi2(2*psi);
    }
    hypothesis_type solve(const ModelType & model, const Eigen::ArrayXi & indices, float tol)
    {
        Eigen::ArrayXf weights = model.get_weights(indices);
        Eigen::ArrayXi order = argsort(weights);
        weights = weights(order);
        Eigen::ArrayXi idx = indices(order).eval();

        const int N = idx.size();

        const int m = model.minimum_set_size();
        const int T_N = niter_RANSAC(p_good_sample, max_outlier_proportion, m, -1);

        int n_star; // termination length (see sec. 2.2 Stopping criterion)
        int I_n_star; // number of inliers found within the first n_star data points
        int I_N_best; // best number of inliers found so far (store the model that goes with it)
        const int I_N_min = (1.-max_outlier_proportion)*N; // the minimum number of total inliers
        int t; // iteration number
        int n; // we draw samples from the set U_n of the top n data points
        double T_n; // average number of samples {M_i}_{i=1}^{T_N} that contain samples from U_n only
        int T_n_prime; // integer version of T_n, see eq. (4)
        int k_n_star; // number of samples to draw to reach the maximality constraint
        int i;
        const double logeta0 = log(eta);

        hypothesis_type p_best;
        Eigen::ArrayXf best_score;

        printf("PROSAC sampling test\n");
        printf("number of correspondences (N):%d\n", N);
        printf("sample size (m):%d\n", m);

        n_star = N;
        I_n_star = 0;
        I_N_best = 0;
        t = 0;
        n = m;
        T_n = T_N;
        for(i=0; i<m; i++) {
            T_n *= (double)(n-i)/(N-i);
        }
        T_n_prime = 1;
        k_n_star = T_N;
        // Note: the condition (I_N_best < I_N_min) was not in the original paper, but it is reasonable:
        // we sholdn't stop if we haven't found the expected number of inliers
        while(((I_N_best < I_N_min) || t <= k_n_star) && t < T_N) {
            // Choice of the hypothesis generation set
            t = t + 1;
            cerr << "t=" << t << endl;
            // from the paper, eq. (5) (not Algorithm1):
            // "The growth function is then deï¬ned as
            //  g(t) = min {n : Tâ€²n â‰¥ t}"
            // Thus n should be incremented if t > T'n, not if t = T'n as written in the algorithm 1
            if ((t > T_n_prime) && (n < n_star)) {
                double T_nplus1 = (T_n * (n+1)) / (n+1-m);
                n = n+1;
                T_n_prime = T_n_prime + ceil(T_nplus1 - T_n);
                //printf("g(t)=n=%d, n_star=%d, T_n-1>=%d, T_n>=%d, T_n'=%d...",
                //    n, n_star, (int)ceil(T_n), (int)ceil(T_nplus1), T_n_prime);
                cerr << "g(t)=" << n << ", n*=" << n_star << ", T_n-1=" << ceil(T_n) << ", T_n>=" << ceil(T_nplus1) << ", T_n'=" << T_n_prime << endl;
                T_n = T_nplus1;
            }
            else {
                //printf("g(t)=n=%d, n_star=%d, T_n>=%d, T_n'=%d: ",
                //    n, n_star, (int)ceil(T_n), T_n_prime);
                cerr << "g(t)=" << n << ", n*=" << n_star << ", T_n-1=" << ceil(T_n) << ", T_n'=" << T_n_prime << endl;
            }
            
            Eigen::ArrayXi sample(m);
            Eigen::ArrayXi sample_indices(m);

            // Draw semi-random sample (note that the test condition from Algorithm1 in the paper is reversed):
            if (t > T_n_prime) {
                // during the finishing stage (n== n_star && t > T_n_prime), draw a standard RANSAC sample
                // The sample contains m points selected from U_n at random
                choice_knuth(n, m, rng, sample);
                cerr << "Draw " << m << " points from U_" << n << endl;
            }
            else {
                // The sample contains m-1 points selected from U_{nâˆ’1} at random and u_n
                choice_knuth(n-1, m-1, rng, sample);
                sample(m-1) = n;
                //printf("Draw %d points from U_%d and point u_%d... ", m-1, n-1, n);
                cerr << "Draw " << m-1 << " points from U_" << n-1 << "and u_" << n << endl;
            }

            sample_indices = idx(sample);

            // INSERT (OPTIONAL): Test for degenerate model configuration (DEGENSAC)
            //                    (i.e. discard the sample if more than 1 model is consistent with the sample)
            // ftp://cmp.felk.cvut.cz/pub/cmp/articles/matas/chum-degen-cvpr05.pdf

            if (!model.sample_check(sample_indices))
            {
                continue;
            }

            // INSERT Compute model parameters p_t from the sample M_t
            //printf("Model parameter estimation... ");
            hypothesis_type p_t = model.fit(sample_indices);

            // Find support of the model with parameters p_t
            // From first paragraph of section 2: "The hypotheses are veriï¬ed against all data"
            auto err = model.error(p_t, idx);
            auto isInlier = (err < tol).eval();
            int I_N = isInlier.count();

            printf("found %d inliers!\n", I_N);

            if (I_N > I_N_best) {
                int n_best; // best value found so far in terms of inliers ratio
                int I_n_best; // number of inliers for n_best

                // INSERT (OPTIONAL): Do local optimization, and recompute the support (LO-RANSAC)
                // http://cmp.felk.cvut.cz/~matas/papers/chum-dagm03.pdf
                // for the fundamental matrix, the normalized 8-points algorithm performs very well:
                // http://axiom.anu.edu.au/~hartley/Papers/fundamental/ICCV-final/fundamental.pdf
                // ...
                // I_N = findSupport(/* model, sample, */ N, isInlier);
                // p_t = model.fit_optimal(isInlier)
                // isInlier = 
                // I_N = 

                
                I_N_best = I_N;

                // INSERT: Store the best model
                p_best = p_t;
                best_score = err;

                // Select new termination length n_star if possible, according to Sec. 2.2.
                // Note: the original paper seems to do it each time a new sample is drawn,
                // but this really makes sense only if the new sample is better than the previous ones.
                n_best = N;
                I_n_best = I_N;

                int n_test; // test value for the termination length
                int I_n_test; // number of inliers for that test value
                double epsilon_n_best = (double)I_n_best/n_best;

                for(n_test = N, I_n_test = I_N; n_test > m; n_test--) { 
                    // Loop invariants:
                    // - I_n_test is the number of inliers for the n_test first correspondences
                    // - n_best is the value between n_test+1 and N that maximizes the ratio I_n_best/n_best
                    assert(n_test >= I_n_test);

                    // * Non-randomness : In >= Imin(n*) (eq. (9))
                    // * Maximality: the number of samples that were drawn so far must be enough
                    // so that the probability of having missed a set of inliers is below eta=0.01.
                    // This is the classical RANSAC termination criterion (HZ 4.7.1.2, eq. (4.18)),
                    // except that it takes into account only the n first samples (not the total number of samples).
                    // kn_star = log(eta0)/log(1-(In_star/n_star)^m) (eq. (12))
                    // We have to minimize kn_star, e.g. maximize I_n_star/n_star
                    //printf("n_best=%d, I_n_best=%d, n_test=%d, I_n_test=%d\n",
                    //        n_best,    I_n_best,    n_test,    I_n_test);
                    // a straightforward implementation would use the following test:
                    //if (I_n_test > epsilon_n_best*n_test) {
                    // However, since In is binomial, and in the case of evenly distributed inliers,
                    // a better test would be to reduce n_star only if there's a significant improvement in
                    // epsilon. Thus we use a Chi-squared test (P=0.10), together with the normal approximation
                    // to the binomial (mu = epsilon_n_star*n_test, sigma=sqrt(n_test*epsilon_n_star*(1-epsilon_n_star)).
                    // There is a significant difference between the two tests (e.g. with the findSupport
                    // functions provided above).
                    // We do the cheap test first, and the expensive test only if the cheap one passes.
                    if (( I_n_test * n_best > I_n_best * n_test ) &&
                        ( I_n_test > epsilon_n_best*n_test + sqrt(n_test*epsilon_n_best*(1.-epsilon_n_best)*2.706) )) {
                        if (I_n_test < Imin(m,n_test)) {
                            // equation 9 not satisfied: no need to test for smaller n_test values anyway
                            break; // jump out of the for(n_test) loop
                        }
                        n_best = n_test;
                        I_n_best = I_n_test;
                        epsilon_n_best = (double)I_n_best/n_best;
                    }

                    // prepare for next loop iteration
                    I_n_test -= isInlier(n_test-1);
                } // for(n_test ...

                // is the best one we found even better than n_star?
                if ( I_n_best * n_star > I_n_star * n_best ) {
                    double logarg;
                    assert(n_best >= I_n_best);
                    // update all values
                    n_star = n_best;
                    I_n_star = I_n_best;
                    k_n_star = niter_RANSAC(1.-eta, 1.-I_n_star/(double)n_star, m, T_N);
                    printf("new values: n_star=%d, k_n_star=%d, I_n_star=%d, I_min=%d\n", n_star, k_n_star, I_n_star, Imin(m,n_best));
                }
            } // if (I_N > I_N_best)
        } // while(t <= k_n_star ...
        //return p_best;
        auto inls = nonzero(best_score<tol);
        return model.fit_optimal(idx(inls));
    }
};


} // namespace