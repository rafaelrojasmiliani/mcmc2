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
 * Verifying the RWMH sampler on a uniform distribution over a hyperrectangle.
 *
 * Motivation for newcomers:
 *   - Axis-aligned hyperrectangles generalise the unit hypercube to allow
 *     dimension-specific bounds, which is a more realistic scenario for
 *     practical models.
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
#include<mcmc/mcmc.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <cstdlib>
#include <utility>
#include <vector>

namespace
{

using Bounds = std::vector<std::pair<double, double>>;

std::optional<double>
log_hyperrectangle_density(const Eigen::VectorXd& values, const Bounds& bounds)
{
    if (bounds.size() != static_cast<std::size_t>(values.size())) {
        return std::nullopt;
    }

    for (Eigen::Index i = 0; i < values.size(); ++i) {
        const double lower = bounds[static_cast<std::size_t>(i)].first;
        const double upper = bounds[static_cast<std::size_t>(i)].second;

        if (values(i) < lower || values(i) > upper) {
            return std::nullopt;
        }
    }

    return 0.0;
}

double
log_hyperrectangle_density_adapter(const Eigen::VectorXd& values, void* data)
{
    if (data == nullptr) {
        return -std::numeric_limits<double>::infinity();
    }

    const auto* bounds = static_cast<const Bounds*>(data);
    const auto maybe_log_density = log_hyperrectangle_density(values, *bounds);
    if (!maybe_log_density) {
        return -std::numeric_limits<double>::infinity();
    }

    return *maybe_log_density;
}

//! Draw iid samples from the uniform distribution on the supplied hyperrectangle.
//!
//! We use the standard library generator so that the reference sample shares
//! the same source of randomness as the MCMC run.  The samples are returned in
//! an n_samples-by-dimension matrix with one draw per row.
Eigen::MatrixXd
draw_uniform_samples(std::mt19937& rng, std::size_t n_samples, const Bounds& bounds)
{
    const int dimension = static_cast<int>(bounds.size());

    Eigen::MatrixXd samples(static_cast<Eigen::Index>(n_samples), dimension);

    std::vector<std::uniform_real_distribution<double>> per_dimension_distributions;
    per_dimension_distributions.reserve(bounds.size());
    for (const auto& bound : bounds) {
        per_dimension_distributions.emplace_back(bound.first, bound.second);
    }

    for (std::size_t i = 0; i < n_samples; ++i) {
        for (int j = 0; j < dimension; ++j) {
            samples(static_cast<Eigen::Index>(i), j) = per_dimension_distributions[static_cast<std::size_t>(j)](rng);
        }
    }

    return samples;
}

//! Compute per-coordinate sample variances given a matrix of draws.
//!
//! The function accepts a pre-computed mean so that callers can keep a single
//! pass over their samples for the summaries they need.  Variances are
//! calculated with Bessel's correction, matching the behaviour of
//! std::sample_variance in statistics texts.
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

    Bounds bounds(static_cast<std::size_t>(dimension));
    std::uniform_real_distribution<double> lower_sampler(-1.0, 0.5);
    std::uniform_real_distribution<double> width_sampler(0.25, 2.0);
    double max_width = 0.0;

    for (int i = 0; i < dimension; ++i) {
        const double lower = lower_sampler(rng);
        const double width = width_sampler(rng);
        const double upper = lower + width;
        bounds[static_cast<std::size_t>(i)] = std::make_pair(lower, upper);
        max_width = std::max(max_width, width);
    }

    const double epsilon = 1.0e-2; // tolerance for the empirical mean
    const double delta = 1.0e-2;   // failure probability in Hoeffding's inequality

    // Hoeffding's inequality bounds the deviation of the empirical mean from
    // the true mean for iid bounded variables.  For a scalar component confined
    // to [a_i, b_i] the tail bound reads
    //   P(|\bar{X}_n - \mathbb{E}[X]| \geq \epsilon) \leq 2\exp\left(-\frac{2n\epsilon^2}{(b_i - a_i)^2}\right).
    // Solving for n shows that drawing at least
    //   n \geq \frac{(b_i - a_i)^2}{2\epsilon^2} \log\left(\frac{2}{\delta}\right)
    // samples keeps that probability below the user-selected \delta.
    // In this general hyperrectangle the worst-case range is the widest side,
    // so we scale the bound by (\max_i b_i - a_i)^2.
    const std::size_t n_samples = std::max<std::size_t>(1, static_cast<std::size_t>(
        std::ceil((max_width * max_width) * std::log(2.0 / delta) / (2.0 * epsilon * epsilon))
    ));

    // Random-Walk Metropolis-Hastings produces a Markov chain that only
    // converges to the target density after an initial transient. Because of
    // this we discard a conservative number of early iterations before
    // measuring any expectations.  We take the burn-in as the larger of two
    // heuristics: (i) 200 iterations per dimension so each coordinate sees a
    // few hundred accepted perturbations before being trusted (with the
    // par_scale chosen above and a 0.3 acceptance rate that covers multiple
    // traversals of the interval), and (ii) ten percent of the requested kept
    // draws so long runs still devote enough time to reaching stationarity.
    const std::size_t n_burnin = std::max<std::size_t>(static_cast<std::size_t>(dimension * 200), n_samples / 10);

    Eigen::VectorXd initial_values(dimension);
    Eigen::VectorXd lower_bounds(dimension);
    Eigen::VectorXd upper_bounds(dimension);

    for (int i = 0; i < dimension; ++i) {
        const double lower = bounds[static_cast<std::size_t>(i)].first;
        const double upper = bounds[static_cast<std::size_t>(i)].second;
        initial_values(i) = 0.5 * (lower + upper);
        lower_bounds(i) = lower;
        upper_bounds(i) = upper;
    }

    mcmc::algo_settings_t settings;
    // RWMH proposes unconstrained Gaussian perturbations; without intervention
    // those jumps could leave the bounded support and force immediate
    // rejections. Because of this we enable the library's automatic truncation
    // so proposals are reflected back into the hyperrectangle.
    settings.vals_bound = true;
    settings.lower_bounds = lower_bounds;
    settings.upper_bounds = upper_bounds;
    // For the same reason—RWMH needs time to forget its initial state before
    // it resembles the stationary distribution—we burn a generous number of
    // draws before we begin accumulating statistics.
    settings.rwmh_settings.n_burnin_draws = n_burnin;
    // Match the direct Monte Carlo sample size for a fair comparison.
    settings.rwmh_settings.n_keep_draws = n_samples;
    // RWMH proposes x_{t+1} = x_t + \eta where \eta ~ N(0, par_scale^2 * cov).
    // Because the perturbation size is controlled by par_scale, picking a
    // moderately small value keeps most proposals inside the hyperrectangle and
    // leads to the textbook ~0.3 acceptance rate for random-walk samplers in
    // moderate dimensions.
    settings.rwmh_settings.par_scale = 0.35;
    // Use an isotropic proposal so that all coordinates share the same scale.
    settings.rwmh_settings.cov_mat = Eigen::MatrixXd::Identity(dimension, dimension);

    Eigen::MatrixXd mcmc_draws;
    // Arguments passed to mcmc::rwmh:
    //   * initial_values seeds the Markov chain at the centre of the hyperrectangle.
    //   * log_hyperrectangle_density_adapter evaluates the log target density
    //     (returning -inf outside the support so those proposals are rejected).
    //   * mcmc_draws receives the kept samples as rows once the run finishes.
    //   * &bounds makes the dimension-specific bounds available to the log-density.
    //   * settings carries all tuning choices discussed above.
    mcmc::rwmh(initial_values, log_hyperrectangle_density_adapter, mcmc_draws, static_cast<void*>(&bounds), settings);

    Eigen::MatrixXd direct_draws = draw_uniform_samples(rng, n_samples, bounds);

    const Eigen::VectorXd mcmc_mean = mcmc_draws.colwise().mean();
    const Eigen::VectorXd direct_mean = direct_draws.colwise().mean();

    const Eigen::VectorXd mcmc_variance = compute_column_variances(mcmc_draws, mcmc_mean);
    const Eigen::VectorXd direct_variance = compute_column_variances(direct_draws, direct_mean);

    const double mean_difference = (mcmc_mean - direct_mean).cwiseAbs().maxCoeff();
    const double variance_difference = (mcmc_variance - direct_variance).cwiseAbs().maxCoeff();

    Eigen::VectorXd theoretical_mean(dimension);
    Eigen::VectorXd theoretical_variance(dimension);

    for (int i = 0; i < dimension; ++i) {
        const double lower = bounds[static_cast<std::size_t>(i)].first;
        const double upper = bounds[static_cast<std::size_t>(i)].second;
        theoretical_mean(i) = 0.5 * (lower + upper);
        const double width = upper - lower;
        theoretical_variance(i) = (width * width) / 12.0;
    }

    std::cout << std::setprecision(6) << std::fixed;
    std::cout << "Uniform hyperrectangle consistency check" << '\n';
    std::cout << "  Dimension (n): " << dimension << '\n';
    std::cout << "  Hoeffding epsilon: " << epsilon << ", delta: " << delta << '\n';
    std::cout << "  Bounds:" << '\n';
    for (int i = 0; i < dimension; ++i) {
        const double lower = bounds[static_cast<std::size_t>(i)].first;
        const double upper = bounds[static_cast<std::size_t>(i)].second;
        std::cout << "    [" << lower << ", " << upper << "]" << '\n';
    }
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
