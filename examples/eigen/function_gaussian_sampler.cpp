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
 * Sampling from a multivariate Gaussian defined as a function f: R^n -> R
 *
 * Context for newcomers:
 *   - Problem: draw samples from a target density specified only through a
 *     log-density function f(x) = log p(x) on R^n.  Here we pick a
 *     multivariate Normal with known mean vector and diagonal covariance to
 *     illustrate how to plug a mathematical description directly into an MCMC
 *     sampler without relying on conjugacy.
 *   - Mathematical target: for x in R^n, the log-density is
 *       f(x) = -\tfrac{1}{2} (x - mu)^T diag(\sigma^{-2}) (x - mu)
 *              - \tfrac{1}{2} \sum_j \log(2\pi\sigma_j^2),
 *     which is the exact Normal log pdf with mean mu and component-wise
 *     standard deviations sigma_j.  We expose both the value of f and its
 *     gradient \nabla f(x) = - diag(\sigma^{-2}) (x - mu) so that gradient-based
 *     samplers can be employed.
 *   - Why the mcmc library: the Hamiltonian Monte Carlo routine handles the
 *     numerical integration and accept-reject mechanics required to simulate
 *     from e^{f(x)}.  We only need to provide f and \nabla f in Eigen types.
 *   - Why Eigen: the state vector lives in R^n.  Eigen arrays/vectors provide a
 *     compact way to evaluate quadratic forms, gradients, and to interoperate
 *     with the library's wrappers.
 */

// $CXX -Wall -std=c++14 -O3 -mcpu=native -ffp-contract=fast -I$EIGEN_INCLUDE_PATH -I./../../include/ function_gaussian_sampler.cpp -o function_gaussian_sampler.out -L./../.. -lmcmc

#define MCMC_ENABLE_EIGEN_WRAPPERS
#include "mcmc.hpp"

#include <cmath>
#include <iostream>
#include <numeric>

struct gaussian_target_data {
    Eigen::VectorXd mean;
    Eigen::VectorXd inv_var;
    double log_normalization;
    int dim;
};

inline double log_gaussian_density(const Eigen::VectorXd& x, Eigen::VectorXd* grad_out, void* data)
{
    gaussian_target_data* params = reinterpret_cast<gaussian_target_data*>(data);

    const Eigen::ArrayXd diff = x.array() - params->mean.array();
    const double quad_form = (diff.square() * params->inv_var.array()).sum();

    if (grad_out) {
        grad_out->resize(params->dim);
        grad_out->array() = -diff * params->inv_var.array();
    }

    return params->log_normalization - 0.5 * quad_form;
}

int main()
{
    const int dim = 3;

    Eigen::VectorXd mean(dim);
    mean << 1.5, -0.5, 2.0;

    Eigen::VectorXd std_dev(dim);
    std_dev << 1.0, 0.6, 1.8;

    const double pi = 3.14159265358979323846;

    gaussian_target_data params;
    params.mean = mean;
    params.inv_var = std_dev.array().square().inverse().matrix();
    params.dim = dim;
    params.log_normalization = -std_dev.array().log().sum()
                              - 0.5 * static_cast<double>(dim) * std::log(2.0 * pi);

    Eigen::VectorXd initial_val = mean + Eigen::VectorXd::Constant(dim, 0.2);

    mcmc::algo_settings_t settings;
    settings.hmc_settings.step_size = 0.15;
    settings.hmc_settings.n_leap_steps = 20;
    settings.hmc_settings.n_burnin_draws = 1500;
    settings.hmc_settings.n_keep_draws = 3000;

    Eigen::MatrixXd draws_out;
    mcmc::hmc(initial_val, log_gaussian_density, draws_out, &params, settings);

    std::cout << "Acceptance rate: "
              << static_cast<double>(settings.hmc_settings.n_accept_draws) / settings.hmc_settings.n_keep_draws
              << std::endl;

    const auto n_draws = static_cast<std::size_t>(draws_out.rows());

    for (int j = 0; j < dim; ++j) {
        const double* begin = draws_out.col(j).data();
        const double* end = begin + n_draws;

        const double sample_mean = std::accumulate(begin, end, 0.0) / static_cast<double>(n_draws);
        const double sq_sum = std::inner_product(begin, end, begin, 0.0);
        const double sample_var = sq_sum / static_cast<double>(n_draws) - sample_mean * sample_mean;
        const double sample_std = std::sqrt(sample_var);

        std::cout << "Dimension " << j << ":\n";
        std::cout << "  sample mean  = " << sample_mean << " (truth " << mean(j)
                  << ")\n";
        std::cout << "  sample std   = " << sample_std << " (truth " << std_dev(j)
                  << ")\n";
    }

    return 0;
}
