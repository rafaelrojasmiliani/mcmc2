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
 * Sampling from a Gaussian distribution using MALA
 *
 * Context for newcomers:
 *   - Problem: estimate the mean and standard deviation of a Normal
 *     distribution when both are unknown.  We build a probabilistic model for
 *     these parameters and wish to draw posterior samples.
 *   - Mathematical target: condition on data x_1,...,x_n ~ Normal(mu, sigma^2)
 *     with an improper prior pi(mu,log sigma) proportional to 1.  The
 *     log-posterior ell(mu,sigma) = -n log sigma - 0.5 sigma^{-2} sum_i (x_i -
 *     mu)^2 provides the score vector used by the Langevin proposal.
 *   - Why the mcmc library: the Metropolis-adjusted Langevin algorithm (MALA)
 *     combines gradient information with random perturbations; this library
 *     provides the carefully tuned proposal mechanics and bookkeeping needed
 *     for the method.
 *   - Why Eigen: gradients, parameter vectors, and proposal perturbations are
 *     stored as Eigen vectors/matrices, letting us express the underlying
 *     calculus in a familiar linear algebra language.
 */

// $CXX -Wall -std=c++14 -O3 -mcpu=native -ffp-contract=fast
// -I$EIGEN_INCLUDE_PATH -I./../../include/ mala_normal.cpp -o mala_normal.out
// -L./../.. -lmcmc

#undef MCMC_ENABLE_EIGEN_WRAPPERS
#define MCMC_ENABLE_EIGEN_WRAPPERS
#include <mcmc/mala.hpp>
#include <mcmc/misc/mcmc_structs.hpp>

inline Eigen::VectorXd eigen_randn_colvec(size_t nr) {
  static std::mt19937 gen{std::random_device{}()};
  static std::normal_distribution<> dist;

  return Eigen::VectorXd{nr}.unaryExpr([&](double x) {
    (void)(x);
    return dist(gen);
  });
}

struct norm_data_t {
  Eigen::VectorXd x;
};

double ll_dens(const Eigen::VectorXd &vals_inp, Eigen::VectorXd *grad_out,
               void *ll_data) {
  const double pi = 3.14159265358979;

  const double mu = vals_inp(0);
  const double sigma = vals_inp(1);

  norm_data_t *dta = reinterpret_cast<norm_data_t *>(ll_data);
  const Eigen::VectorXd x = dta->x;
  const int n_vals = x.size();

  //

  const double ret = -n_vals * (0.5 * std::log(2 * pi) + std::log(sigma)) -
                     (x.array() - mu).pow(2).sum() / (2 * sigma * sigma);

  //

  if (grad_out) {
    grad_out->resize(2, 1);

    //

    const double m_1 = (x.array() - mu).sum();
    const double m_2 = (x.array() - mu).pow(2).sum();

    (*grad_out)(0, 0) = m_1 / (sigma * sigma);
    (*grad_out)(1, 0) =
        (m_2 / (sigma * sigma * sigma)) - ((double)n_vals) / sigma;
  }

  //

  return ret;
}

double log_target_dens(const Eigen::VectorXd &vals_inp,
                       Eigen::VectorXd *grad_out, void *ll_data) {
  return ll_dens(vals_inp, grad_out, ll_data);
}

int main() {
  const int n_data = 1000;

  const double mu = 2.0;
  const double sigma = 2.0;

  norm_data_t dta;

  Eigen::VectorXd x_dta = mu + sigma * eigen_randn_colvec(n_data).array();
  dta.x = x_dta;

  Eigen::VectorXd initial_val(2);
  initial_val(0) = mu + 1;    // mu
  initial_val(1) = sigma + 1; // sigma

  mcmc::algo_settings_t settings;

  settings.mala_settings.step_size = 0.08;
  settings.mala_settings.n_burnin_draws = 2000;
  settings.mala_settings.n_keep_draws = 2000;

  //

  Eigen::MatrixXd draws_out;
  mcmc::mala(initial_val, log_target_dens, draws_out, &dta, settings);

  //

  std::cout << "mala mean:\n" << draws_out.colwise().mean() << std::endl;
  std::cout << "acceptance rate: "
            << static_cast<double>(settings.mala_settings.n_accept_draws) /
                   settings.mala_settings.n_keep_draws
            << std::endl;

  //

  return 0;
}
