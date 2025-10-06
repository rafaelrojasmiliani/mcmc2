/*################################################################################
  ##
  ##   Copyright (C) 2011-2023 Keith O'Hara
  ##
  ##   This file is part of the MCMC C++ library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
  ##
  ################################################################################*/

/*
 * Verifying the RWMH sampler on a uniform distribution over the unit hypercube.
 *
 * Motivation for newcomers:
 *   - The unit hypercube [0, 1]^n is a prototypical bounded target density.
 *   - MCMC samplers that implement automatic bounds handling (such as the
 *     Random-Walk Metropolis-Hastings routine in MCMCLib) should be able to
 *     recover the same summary statistics as direct sampling with
 *     std::uniform_real_distribution.
 *   - We choose the ambient dimension n uniformly between 1 and 10, then use
 *     Hoeffding's inequality to obtain a conservative sample size that keeps
 *     Monte Carlo error for the sample means within a user-selected tolerance.
 *   - With that number of samples we compare the empirical means and variances
 *     produced by MCMCLib and by the standard C++ generator.
 */

// Example compile command (requires Eigen headers in the include path):
// $CXX -Wall -std=c++14 -O3 -march=native -ffp-contract=fast -I$EIGEN_INCLUDE_PATH -I./../../include/ \
//     uniform_hypercube.cpp -o uniform_hypercube.out -L./../.. -lmcmc

#define MCMC_ENABLE_EIGEN_WRAPPERS
#include "mcmc.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <cstdlib>

namespace
{

double
log_unit_hypercube_density(const Eigen::VectorXd& values, void*)
{
    if ((values.array() < 0.0).any() || (values.array() > 1.0).any()) {
        return -std::numeric_limits<double>::infinity();
    }

    return 0.0;
}

Eigen::MatrixXd
draw_uniform_samples(std::mt19937& rng, std::size_t n_samples, int dimension)
{
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    Eigen::MatrixXd samples(static_cast<Eigen::Index>(n_samples), dimension);

    for (std::size_t i = 0; i < n_samples; ++i) {
        for (int j = 0; j < dimension; ++j) {
            samples(static_cast<Eigen::Index>(i), j) = uniform01(rng);
        }
    }

    return samples;
}

Eigen::VectorXd
compute_column_variances(const Eigen::MatrixXd& samples, const Eigen::VectorXd& mean)
{
    Eigen::VectorXd variances(samples.cols());
    variances.setZero();

    const Eigen::Index n_rows = samples.rows();

    if (n_rows <= 1) {
        return variances;
    }

    const Eigen::MatrixXd centered = samples.rowwise() - mean.transpose();
    variances = (centered.array().square().colwise().sum() / static_cast<double>(n_rows - 1)).matrix();

    return variances;
}

} // namespace

int
main()
{
    std::random_device rd;
    std::mt19937 rng(rd());

    std::uniform_int_distribution<int> dimension_sampler(1, 10);
    const int dimension = dimension_sampler(rng);

    const double epsilon = 1.0e-2; // tolerance for the empirical mean
    const double delta = 1.0e-2;   // failure probability in Hoeffding's inequality

    const std::size_t n_samples = static_cast<std::size_t>(
        std::ceil(std::log(2.0 / delta) / (2.0 * epsilon * epsilon))
    );

    const std::size_t n_burnin = std::max<std::size_t>(static_cast<std::size_t>(dimension * 200), n_samples / 10);

    Eigen::VectorXd initial_values = Eigen::VectorXd::Constant(dimension, 0.5);

    mcmc::algo_settings_t settings;
    settings.vals_bound = true;
    settings.lower_bounds = Eigen::VectorXd::Zero(dimension);
    settings.upper_bounds = Eigen::VectorXd::Ones(dimension);
    settings.rwmh_settings.n_burnin_draws = n_burnin;
    settings.rwmh_settings.n_keep_draws = n_samples;
    settings.rwmh_settings.par_scale = 0.35;
    settings.rwmh_settings.cov_mat = Eigen::MatrixXd::Identity(dimension, dimension);

    Eigen::MatrixXd mcmc_draws;
    mcmc::rwmh(initial_values, log_unit_hypercube_density, mcmc_draws, nullptr, settings);

    Eigen::MatrixXd direct_draws = draw_uniform_samples(rng, n_samples, dimension);

    const Eigen::VectorXd mcmc_mean = mcmc_draws.colwise().mean();
    const Eigen::VectorXd direct_mean = direct_draws.colwise().mean();

    const Eigen::VectorXd mcmc_variance = compute_column_variances(mcmc_draws, mcmc_mean);
    const Eigen::VectorXd direct_variance = compute_column_variances(direct_draws, direct_mean);

    const double mean_difference = (mcmc_mean - direct_mean).cwiseAbs().maxCoeff();
    const double variance_difference = (mcmc_variance - direct_variance).cwiseAbs().maxCoeff();

    const Eigen::VectorXd theoretical_mean = Eigen::VectorXd::Constant(dimension, 0.5);
    const Eigen::VectorXd theoretical_variance = Eigen::VectorXd::Constant(dimension, 1.0 / 12.0);

    std::cout << std::setprecision(6) << std::fixed;
    std::cout << "Uniform hypercube consistency check" << '\n';
    std::cout << "  Dimension (n): " << dimension << '\n';
    std::cout << "  Hoeffding epsilon: " << epsilon << ", delta: " << delta << '\n';
    std::cout << "  Burn-in draws: " << n_burnin << '\n';
    std::cout << "  Kept draws per method: " << n_samples << '\n';
    std::cout << "  MCMC acceptance rate: "
              << static_cast<double>(settings.rwmh_settings.n_accept_draws) / static_cast<double>(settings.rwmh_settings.n_keep_draws)
              << '\n';
    std::cout << "  MCMC mean: " << mcmc_mean.transpose() << '\n';
    std::cout << "  Direct mean: " << direct_mean.transpose() << '\n';
    std::cout << "  Mean difference (L-inf): " << mean_difference << '\n';
    std::cout << "  MCMC variance: " << mcmc_variance.transpose() << '\n';
    std::cout << "  Direct variance: " << direct_variance.transpose() << '\n';
    std::cout << "  Variance difference (L-inf): " << variance_difference << '\n';
    std::cout << "  Theoretical mean: " << theoretical_mean.transpose() << '\n';
    std::cout << "  Theoretical variance: " << theoretical_variance.transpose() << '\n';

    const double mean_tolerance = 2.0 * epsilon;
    const double variance_tolerance = 5.0 * epsilon;

    if (mean_difference > mean_tolerance || variance_difference > variance_tolerance) {
        std::cerr << "Consistency check failed: sample summaries differ beyond tolerance." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Consistency check passed within tolerances." << std::endl;
    return EXIT_SUCCESS;
}
