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
 *     standard deviations sigma_j.  Working with log p(x) rather than p(x)
 *     avoids numerical underflow for high-dimensional states and makes
 *     constant factors in p(x) irrelevant because MCMC only requires ratios
 *     p(x') / p(x) = exp(f(x') - f(x)).  We expose both the value of f and its
 *     gradient \nabla f(x) = - diag(\sigma^{-2}) (x - mu) so that
 * gradient-based samplers can be employed.
 *   - Why the mcmc library: the Hamiltonian Monte Carlo routine handles the
 *     numerical integration and accept-reject mechanics required to simulate
 *     from e^{f(x)}.  We only need to provide f and \nabla f in Eigen types.
 *   - Why Eigen: the state vector lives in R^n.  Eigen arrays/vectors provide a
 *     compact way to evaluate quadratic forms, gradients, and to interoperate
 *     with the library's wrappers.
 *   - Workflow with this example:
 *       1. define distribution parameters (mu, sigma) and package them in
 *          gaussian_target_data so log_gaussian_density can evaluate f(x).
 *       2. initiate algo_settings_t with HMC hyperparameters and an initial
 *          state x_0.
 *       3. call mcmc::hmc, which repeatedly invokes log_gaussian_density to
 *          obtain log-values and gradients while producing Monte Carlo draws.
 */

// $CXX -Wall -std=c++14 -O3 -mcpu=native -ffp-contract=fast
// -I$EIGEN_INCLUDE_PATH -I./../../include/ function_gaussian_sampler.cpp -o
// function_gaussian_sampler.out -L./../.. -lmcmc

#define MCMC_ENABLE_EIGEN_WRAPPERS
#include <mcmc/hmc.hpp>
#include <mcmc/misc/mcmc_structs.hpp>

#include <cmath>
#include <iostream>
#include <numeric>

/*
 * Container holding the parameters that define the log-target f(x):
 *   mean             mu in R^n.
 *   inv_var          diagonal entries of Sigma^{-1} written as variances^{-1}.
 *   log_normalization  -0.5 n log(2pi) - sum_j log sigma_j, the constant part
 * of f. dim              the ambient dimension n used to size gradients. These
 * values are passed through the generic `void* data` channel expected by the
 * mcmc library's function-based samplers so that our log-density has access to
 * the data it needs without requiring global state.
 */
struct gaussian_target_data {
  Eigen::VectorXd mean;
  Eigen::VectorXd inv_var;
  double log_normalization;
  int dim;
};

/*
 * log_gaussian_density(x, grad_out, data)
 *   Relation to the target: returns f(x) = log p(x) for the Gaussian density
 * and, if requested, populates grad_out with the gradient \nabla f(x) =
 * -Sigma^{-1}(x - mu). Inputs:
 *     - x: Eigen vector representing the current state in R^n.
 *     - grad_out: optional Eigen vector pointer supplied by the MCMC routine;
 * when non-null it must be resized and filled with \nabla f(x).
 *     - data: raw pointer forwarded by the sampler; we reinterpret_cast it back
 * to gaussian_target_data so the function can read mu, Sigma^{-1}, and
 * constants. Output:
 *     - returns the scalar log-density f(x) used by HMC in its Hamiltonian.
 *   Signature rationale:
 *     The mcmc library expects log-target callbacks of type
 *       double(const Eigen::VectorXd&, Eigen::VectorXd*, void*).
 *     This uniform interface lets the library wrap different targets without
 *     templates, while still supporting gradient-aware algorithms when grad_out
 *     is provided.  When grad_out is nullptr the function should skip gradient
 *     work, mirroring the fact that gradient-free samplers only require f(x).
 */
inline double log_gaussian_density(const Eigen::VectorXd &x,
                                   Eigen::VectorXd *grad_out, void *data) {
  gaussian_target_data *params = reinterpret_cast<gaussian_target_data *>(data);

  const Eigen::ArrayXd diff = x.array() - params->mean.array();
  const double quad_form = (diff.square() * params->inv_var.array()).sum();

  if (grad_out) {
    grad_out->resize(params->dim);
    grad_out->array() = -diff * params->inv_var.array();
  }

  return params->log_normalization - 0.5 * quad_form;
}

int main() {
  // Step 1: define distribution parameters mu and sigma, encoding the target
  // density p(x) = N(mu, diag(sigma^2)).  These vectors fully specify the
  // quadratic form (x - mu)^T diag(sigma^{-2})(x - mu) and hence f(x).
  const int dim = 3;

  Eigen::VectorXd mean(dim);
  mean << 1.5, -0.5, 2.0;

  Eigen::VectorXd std_dev(dim);
  std_dev << 1.0, 0.6, 1.8;

  const double pi = 3.14159265358979323846;

  // Package the parameters and normalization constant so log_gaussian_density
  // can evaluate f(x) = log p(x) and \nabla f(x) when invoked by the sampler.
  gaussian_target_data params;
  params.mean = mean;
  params.inv_var = std_dev.array().square().inverse().matrix();
  params.dim = dim;
  params.log_normalization =
      -std_dev.array().log().sum() -
      0.5 * static_cast<double>(dim) * std::log(2.0 * pi);

  // Step 2: choose an initial state x_0 and configure algorithmic settings.
  // The initial point seeds the Hamiltonian dynamics, while the settings fix
  // how we discretize Hamilton's equations for the separable Hamiltonian
  //   H(x, p) = -f(x) + 0.5 * ||p||^2
  // where auxiliary momenta p ~ N(0, I) are introduced at each iteration.
  Eigen::VectorXd initial_val = mean + Eigen::VectorXd::Constant(dim, 0.2);

  mcmc::algo_settings_t settings;
  // The step size epsilon controls the leapfrog discretization of the coupled
  // ODEs d x / d t = +\partial H / \partial p = p and d p / d t = -\partial H /
  // \partial x = \nabla f(x); smaller epsilon better approximates the
  // continuous flow but costs more gradient evaluations for a fixed trajectory
  // length.
  settings.hmc_settings.step_size = 0.15;
  // L leapfrog updates advance (x, p) by roughly L * epsilon units of simulated
  // time, setting the proposal distance in phase space before the Metropolis
  // correction accepts/rejects.
  settings.hmc_settings.n_leap_steps = 20;
  // Draws produced while the chain forgets its arbitrary x_0 are discarded to
  // remove dependence on initialization (burn-in / transient phase).
  settings.hmc_settings.n_burnin_draws = 1500;
  // The remaining draws approximate expectations E[g(X)] via Monte Carlo
  // averages.
  settings.hmc_settings.n_keep_draws = 3000;

  // Step 3: call the Hamiltonian Monte Carlo driver with our callback f.
  // HMC simulates Hamiltonian flow using gradients from log_gaussian_density
  // to propose distant yet high-probability states, then applies a Metropolis
  // correction using differences f(x') - f(x).
  Eigen::MatrixXd draws_out;
  // Inputs: initial_val is x_0 in R^n, log_gaussian_density supplies log p(x)
  // and \nabla log p(x), draws_out stores the retained samples row-wise, params
  // passes mu and Sigma information through the void* interface, and settings
  // encapsulates the HMC hyperparameters set above.
  mcmc::hmc(initial_val, log_gaussian_density, draws_out, &params, settings);

  std::cout << "Acceptance rate: "
            << static_cast<double>(settings.hmc_settings.n_accept_draws) /
                   settings.hmc_settings.n_keep_draws
            << std::endl;

  // Diagnostic check: (i) treat the rows of draws_out as samples X^{(s)},
  // (ii) compute the empirical mean \bar{X}_j and variance for each coordinate,
  // (iii) compare them with the true Normal parameters to assess convergence.
  const auto n_draws = static_cast<std::size_t>(draws_out.rows());

  for (int j = 0; j < dim; ++j) {
    const double *begin = draws_out.col(j).data();
    const double *end = begin + n_draws;

    // Step (ii-a): compute \bar{X}_j = (1 / S) \sum_s X^{(s)}_j via accumulate.
    const double sample_mean =
        std::accumulate(begin, end, 0.0) / static_cast<double>(n_draws);
    // Step (ii-b): evaluate the second moment (1 / S) \sum_s (X^{(s)}_j)^2
    // using inner_product.
    const double sq_sum = std::inner_product(begin, end, begin, 0.0);
    // Step (ii-c): combine the first and second moments to obtain Var[X_j].
    const double sample_var =
        sq_sum / static_cast<double>(n_draws) - sample_mean * sample_mean;
    const double sample_std = std::sqrt(sample_var);

    std::cout << "Dimension " << j << ":\n";
    std::cout << "  sample mean  = " << sample_mean << " (truth " << mean(j)
              << ")\n";
    std::cout << "  sample std   = " << sample_std << " (truth " << std_dev(j)
              << ")\n";
  }

  return 0;
}
