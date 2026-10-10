// dryrun/component/rollout/engine_cpp/src/cost_model.cc
#include "engine/cost_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace dryrun::engine {

// --- advance.py closed-form math ---

int crossing_step(double F, double alpha, double beta, int k_cap) {
  int k0;
  if (beta <= 0) {
    k0 = (alpha >= F) ? 0 : (k_cap >= 0 ? k_cap : 0);
  } else if (alpha >= F) {
    k0 = 0;
  } else {
    double need = F - alpha;
    k0 = std::max(0, static_cast<int>(std::ceil(need / beta)));
  }
  if (k_cap >= 0) {
    k0 = std::min(k0, k_cap);
  }
  return k0;
}

double elapsed(double F, double alpha, double beta, int k) {
  if (k <= 0) return 0.0;
  int k0 = crossing_step(F, alpha, beta, /*k_cap=*/k);
  double total = static_cast<double>(k0) * F;
  int n = k - k0;
  if (n > 0) {
    // tri = n * (n - 1) / 2, using integer division.
    long long tri = static_cast<long long>(n) * (n - 1) / 2;
    total += static_cast<double>(n) * (alpha + beta * k0) +
             beta * static_cast<double>(tri);
  }
  return total;
}

int steps_within(double F, double alpha, double beta, double budget) {
  if (budget <= 0) return 0;
  if (beta <= 0) {
    double tau = std::max(F, alpha);
    if (tau <= 0) return 0;
    int k = static_cast<int>(budget / tau);
    // Float division can land one step short when budget is an exact
    // multiple of tau. Step forward while elapsed fits, then step back.
    while (elapsed(F, alpha, beta, k + 1) <= budget) {
      ++k;
    }
    while (k > 0 && elapsed(F, alpha, beta, k) > budget) {
      --k;
    }
    return k;
  }

  int k0 = crossing_step(F, alpha, beta);
  double t_floor = static_cast<double>(k0) * F;
  if (budget <= t_floor) {
    return (F > 0) ? static_cast<int>(budget / F) : k0;
  }

  double rest = budget - t_floor;
  double a = beta / 2.0;
  double b_coeff = alpha + beta * k0 - beta / 2.0;
  double c = -rest;
  double disc = b_coeff * b_coeff - 4.0 * a * c;
  int n = (disc < 0) ? 0
                     : static_cast<int>((-b_coeff + std::sqrt(disc)) / (2.0 * a));
  n = std::max(0, n);
  while (elapsed(F, alpha, beta, k0 + n + 1) <= budget) {
    ++n;
  }
  while (n > 0 && elapsed(F, alpha, beta, k0 + n) > budget) {
    --n;
  }
  return k0 + n;
}

std::pair<int, int> next_completion(const std::vector<int>& remaining) {
  if (remaining.empty()) return {0, 0};
  int k = *std::min_element(remaining.begin(), remaining.end());
  int count = 0;
  for (int r : remaining) {
    if (r == k) ++count;
  }
  return {k, count};
}

// --- RolloutCostModel base ---

double RolloutCostModel::step_time(double n_tokens, double t2, double ctxsum,
                                   int n_reqs, double prefill_ctx) {
  // Infer phase from arguments, matching RolloutWorkload.from_step.
  std::string phase;
  if (t2 > 0 && ctxsum > 0) {
    phase = "mixed";
  } else if (t2 > 0) {
    phase = "prefill";
  } else {
    phase = "decode";
  }
  return predict(phase, n_tokens, t2, ctxsum, n_reqs, prefill_ctx);
}

bool RolloutCostModel::saturated(double /*n_tokens*/, double /*t2*/,
                                 double /*ctxsum*/, int /*n_reqs*/) {
  return false;
}

double RolloutCostModel::decode_segment_time(int n_d, double ctxsum, int k) {
  auto [floor, alpha, beta] = decode_coeffs(n_d, ctxsum);
  return elapsed(floor, alpha, beta, k);
}

int RolloutCostModel::decode_steps_within(int n_d, double ctxsum,
                                          double budget) {
  auto [floor, alpha, beta] = decode_coeffs(n_d, ctxsum);
  return steps_within(floor, alpha, beta, budget);
}

int RolloutCostModel::decode_saturated_steps(int n_d, double ctxsum, int k) {
  auto [floor, alpha, beta] = decode_coeffs(n_d, ctxsum);
  return k - crossing_step(floor, alpha, beta, /*k_cap=*/k);
}

// --- PsrlCostModel ---

PsrlCostModel::PsrlCostModel(double prefill_attn_b, double prefill_attn_k,
                             double prefill_other_floor,
                             double prefill_other_b, double prefill_other_k,
                             double decode_attn_b, double decode_attn_k,
                             double decode_other_floor, double decode_other_b,
                             double decode_other_k)
    : prefill_attn_b_(prefill_attn_b),
      prefill_attn_k_(prefill_attn_k),
      prefill_other_floor_(prefill_other_floor),
      prefill_other_b_(prefill_other_b),
      prefill_other_k_(prefill_other_k),
      decode_attn_b_(decode_attn_b),
      decode_attn_k_(decode_attn_k),
      decode_other_floor_(decode_other_floor),
      decode_other_b_(decode_other_b),
      decode_other_k_(decode_other_k) {
  if (prefill_attn_b < 0 || prefill_attn_k < 0 || prefill_other_floor < 0 ||
      prefill_other_b < 0 || prefill_other_k < 0 || decode_attn_b < 0 ||
      decode_attn_k < 0 || decode_other_floor < 0 || decode_other_b < 0 ||
      decode_other_k < 0) {
    throw std::invalid_argument("PSRL coefficients must be non-negative.");
  }
}

double PsrlCostModel::predict(const std::string& phase, double n_tokens,
                              double sum_l2, double ctxsum, int n_reqs,
                              double /*prefill_ctx*/) {
  double latency;
  if (phase == "prefill") {
    double attention = prefill_attn_b_ + prefill_attn_k_ * sum_l2;
    double linear = prefill_other_b_ + prefill_other_k_ * n_tokens;
    double other = std::max(prefill_other_floor_, linear);
    latency = attention + other;
  } else if (phase == "decode") {
    double attention = decode_attn_b_ + decode_attn_k_ * ctxsum;
    double linear = decode_other_b_ + decode_other_k_ * n_reqs;
    double other = std::max(decode_other_floor_, linear);
    latency = attention + other;
  } else {
    // Mixed step: prefill_attention + decode_attention + max(prefill_other,
    // decode_other).
    double prefill_attention = prefill_attn_b_ + prefill_attn_k_ * sum_l2;
    double prefill_linear =
        prefill_other_b_ + prefill_other_k_ * n_tokens;
    double prefill_other = std::max(prefill_other_floor_, prefill_linear);
    double decode_attention = decode_attn_b_ + decode_attn_k_ * ctxsum;
    double decode_linear =
        decode_other_b_ + decode_other_k_ * n_reqs;
    double decode_other = std::max(decode_other_floor_, decode_linear);
    latency =
        prefill_attention + decode_attention + std::max(prefill_other, decode_other);
  }
  if (latency <= 0) {
    throw std::runtime_error(
        "A rollout cost model must predict positive latency.");
  }
  return latency;
}

std::tuple<double, double, double> PsrlCostModel::decode_coeffs(
    int n_d, double ctxsum) {
  double other = std::max(decode_other_floor_,
                          decode_other_b_ + decode_other_k_ * n_d);
  double alpha = decode_attn_b_ + decode_attn_k_ * ctxsum + other;
  double beta = decode_attn_k_ * n_d;
  return {0.0, alpha, beta};
}

bool PsrlCostModel::saturated(double n_tokens, double t2, double ctxsum,
                               int n_reqs) {
  // Infer phase.
  std::string phase;
  if (t2 > 0 && ctxsum > 0) {
    phase = "mixed";
  } else if (t2 > 0) {
    phase = "prefill";
  } else {
    phase = "decode";
  }

  if (phase == "prefill") {
    return prefill_other_b_ + prefill_other_k_ * n_tokens >=
           prefill_other_floor_;
  }
  if (phase == "decode") {
    return decode_other_b_ + decode_other_k_ * n_reqs >= decode_other_floor_;
  }
  // Mixed.
  return (prefill_other_b_ + prefill_other_k_ * n_tokens >=
          prefill_other_floor_) &&
         (decode_other_b_ + decode_other_k_ * n_reqs >= decode_other_floor_);
}

// --- UnifiedRooflineCostModel ---

UnifiedRooflineCostModel::UnifiedRooflineCostModel(double F, double W,
                                                   double G, double A_p,
                                                   double A_d, int b)
    : F_(F), W_(W), G_(G), A_p_(A_p), A_d_(A_d), b_(b) {
  if (b <= 0) {
    throw std::invalid_argument("b must be positive.");
  }
  if (F < 0 || W < 0 || G < 0 || A_p < 0 || A_d < 0) {
    throw std::invalid_argument(
        "Roofline coefficients must be non-negative.");
  }
}

double UnifiedRooflineCostModel::raw(const std::string& phase, double n_tokens,
                                     double sum_l2, double ctxsum) const {
  double prefill_sum_l2 =
      (phase == "prefill" || phase == "mixed") ? sum_l2 : 0.0;
  double decode_ctxsum =
      (phase == "decode" || phase == "mixed") ? ctxsum : 0.0;
  return W_ + G_ * n_tokens + A_p_ * prefill_sum_l2 / b_ + A_d_ * decode_ctxsum;
}

double UnifiedRooflineCostModel::predict(const std::string& phase,
                                         double n_tokens, double sum_l2,
                                         double ctxsum, int /*n_reqs*/,
                                         double /*prefill_ctx*/) {
  double r = raw(phase, n_tokens, sum_l2, ctxsum);
  double latency = (r < F_) ? F_ : r;
  if (latency <= 0) {
    throw std::runtime_error(
        "A rollout cost model must predict positive latency.");
  }
  return latency;
}

std::tuple<double, double, double> UnifiedRooflineCostModel::decode_coeffs(
    int n_d, double ctxsum) {
  double alpha = W_ + G_ * n_d + A_d_ * ctxsum;
  double beta = A_d_ * n_d;
  return {F_, alpha, beta};
}

bool UnifiedRooflineCostModel::saturated(double n_tokens, double t2,
                                         double ctxsum, int /*n_reqs*/) {
  std::string phase;
  if (t2 > 0 && ctxsum > 0) {
    phase = "mixed";
  } else if (t2 > 0) {
    phase = "prefill";
  } else {
    phase = "decode";
  }
  return raw(phase, n_tokens, t2, ctxsum) >= F_;
}

}  // namespace dryrun::engine
