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
 * Sampling from a Gaussian distribution using RWMH
 *
 * Context for newcomers:
 *   - Problem: infer the unknown mean of a Normal distribution when the
 *     variance is known and we have a conjugate Normal prior on the mean.
 *   - Mathematical target: likelihood L(mu) proportional to exp(-0.5 sigma^{-2}
 *     sum_i (x_i - mu)^2) with prior mu ~ Normal(mu_0, sigma_0^2).  The
 *     resulting posterior is Normal(mu_n, sigma_n^2) where sigma_n^{-2} =
 *     n/sigma^2 + 1/sigma_0^2 and mu_n = sigma_n^2 (sum_i x_i / sigma^2 +
 *     mu_0 / sigma_0^2), but we purposely simulate it via Metropolis to expose
 *     the algorithmic workflow.
 *   - Why the mcmc library: the Random-Walk Metropolis-Hastings (RWMH)
 *     algorithm implements accept/reject logic and tuning parameters for the
 *     proposal kernel, sparing us from writing boilerplate control flow.
 *   - Why Eigen: even in one dimension, the sampler expects parameter vectors
 *     and data containers compatible with Eigen so that the same interface
 *     works in higher-dimensional models.
 */

// $CXX -Wall -std=c++14 -O3 -mcpu=native -ffp-contract=fast
// -I$EIGEN_INCLUDE_PATH -I./../../include/ rwmh_normal_mean.cpp -o
// rwmh_normal_mean.out -L./../.. -lmcmc

#define MCMC_ENABLE_EIGEN_WRAPPERS
#include <mcmc/misc/mcmc_structs.hpp>
#include <mcmc/rwmh.hpp>

inline Eigen::VectorXd eigen_randn_colvec(size_t nr) {
  static std::mt19937 gen{std::random_device{}()};
  static std::normal_distribution<> dist;

  return Eigen::VectorXd{nr}.unaryExpr([&](double x) {
    (void)(x);
    return dist(gen);
  });
}

struct norm_data_t {
  double sigma;
  Eigen::VectorXd x;

  double mu_0;
  double sigma_0;
};

double ll_dens(const Eigen::VectorXd &vals_inp, void *ll_data) {
  const double pi = 3.14159265358979;

  //

  const double mu = vals_inp(0);

  norm_data_t *dta = reinterpret_cast<norm_data_t *>(ll_data);
  const double sigma = dta->sigma;
  const Eigen::VectorXd x = dta->x;

  const int n_vals = x.size();

  //

  const double ret = -n_vals * (0.5 * std::log(2 * pi) + std::log(sigma)) -
                     (x.array() - mu).pow(2).sum() / (2 * sigma * sigma);

  //

  return ret;
}

double log_pr_dens(const Eigen::VectorXd &vals_inp, void *ll_data) {
  const double pi = 3.14159265358979;

  //

  norm_data_t *dta = reinterpret_cast<norm_data_t *>(ll_data);

  const double mu_0 = dta->mu_0;
  const double sigma_0 = dta->sigma_0;

  const double x = vals_inp(0);

  const double ret = -0.5 * std::log(2 * pi) - std::log(sigma_0) -
                     std::pow(x - mu_0, 2) / (2 * sigma_0 * sigma_0);

  return ret;
}

double log_target_dens(const Eigen::VectorXd &vals_inp, void *ll_data) {
  return ll_dens(vals_inp, ll_data) + log_pr_dens(vals_inp, ll_data);
}

int main() {
  const int n_data = 100;
  const double mu = 2.0;

  norm_data_t dta;
  dta.sigma = 1.0;
  dta.mu_0 = 1.0;
  dta.sigma_0 = 2.0;

  Eigen::VectorXd x_dta = mu + eigen_randn_colvec(n_data).array();
  dta.x = x_dta;

  Eigen::VectorXd initial_val(1);
  initial_val(0) = 1.0;

  //

  mcmc::algo_settings_t settings;

  settings.rwmh_settings.par_scale = 0.4;
  settings.rwmh_settings.n_burnin_draws = 2000;
  settings.rwmh_settings.n_keep_draws = 2000;

  //

  Eigen::MatrixXd draws_out;
  mcmc::rwmh(initial_val, log_target_dens, draws_out, &dta, settings);

  //

  std::cout << "rwmh mean:\n" << draws_out.colwise().mean() << std::endl;
  std::cout << "acceptance rate: "
            << static_cast<double>(settings.rwmh_settings.n_accept_draws) /
                   settings.rwmh_settings.n_keep_draws
            << std::endl;

  //

  return 0;
}
