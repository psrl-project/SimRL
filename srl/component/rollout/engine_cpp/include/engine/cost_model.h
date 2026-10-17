// srl/component/rollout/engine_cpp/include/engine/cost_model.h
//
// Rollout cost model interface and PSRL implementation.
//
// Port of srl/cost/rollout/base.py (RolloutCostModel ABC) and
// srl/cost/rollout/psrl.py (PSRLFitted). Also includes the closed-form
// decode segment math from srl/cost/rollout/advance.py.
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace srl::engine {

// --- Closed-form decode segment math (advance.py) ---

// First step index at which the linear branch overtakes the floor F.
int crossing_step(double F, double alpha, double beta, int k_cap = -1);

// Total time for k steps of tau(i) = max(F, alpha + beta * i).
double elapsed(double F, double alpha, double beta, int k);

// Largest k with elapsed(k) <= budget.
int steps_within(double F, double alpha, double beta, double budget);

// Steps until the earliest completion, and how many finish simultaneously.
std::pair<int, int> next_completion(const std::vector<int>& remaining);

// --- Cost model interface ---

class RolloutCostModel {
 public:
  virtual ~RolloutCostModel() = default;

  // Predict strictly positive wall-clock latency in seconds.
  //
  // Args:
  //   phase: "prefill", "decode", or "mixed".
  //   n_tokens: Total tokens in the step.
  //   sum_l2: Sum of squared prefill extents.
  //   ctxsum: Sum of running-request contexts.
  //   n_reqs: Number of requests in the step.
  //   prefill_ctx: Sum of attention extents of prefilling requests.
  virtual double predict(const std::string& phase, double n_tokens,
                         double sum_l2, double ctxsum, int n_reqs,
                         double prefill_ctx) = 0;

  // Adapt the simulator's compact step arguments to predict.
  //
  // Virtual because a Python model may override it rather than `predict`, and a
  // wrapped model must reach that override. The default forwards to `predict`.
  virtual double step_time(double n_tokens, double t2, double ctxsum,
                           int n_reqs = 0, double prefill_ctx = 0.0);

  // Return floor, intercept, and slope for a pure-decode segment.
  virtual std::tuple<double, double, double> decode_coeffs(int n_d,
                                                           double ctxsum) = 0;

  // Return whether the workload is above a fitted low-load floor.
  virtual bool saturated(double n_tokens, double t2, double ctxsum,
                         int n_reqs = 0);

  // The segment queries below are virtual for the same reason. Their defaults
  // rebuild each answer from `decode_coeffs`, a secant approximation of the
  // curve a model such as `ComposedRollout` integrates exactly.

  // Wall-clock seconds of k decode steps over a fixed batch.
  virtual double decode_segment_time(int n_d, double ctxsum, int k);

  // Largest step count whose segment time fits inside budget.
  virtual int decode_steps_within(int n_d, double ctxsum, double budget);

  // How many of k steps run above the low-load floor.
  virtual int decode_saturated_steps(int n_d, double ctxsum, int k);
};

// --- PSRL fitted model ---

class PsrlCostModel : public RolloutCostModel {
 public:
  PsrlCostModel(double prefill_attn_b, double prefill_attn_k,
                double prefill_other_floor, double prefill_other_b,
                double prefill_other_k, double decode_attn_b,
                double decode_attn_k, double decode_other_floor,
                double decode_other_b, double decode_other_k);

  double predict(const std::string& phase, double n_tokens, double sum_l2,
                 double ctxsum, int n_reqs, double prefill_ctx) override;

  std::tuple<double, double, double> decode_coeffs(int n_d,
                                                   double ctxsum) override;

  bool saturated(double n_tokens, double t2, double ctxsum,
                 int n_reqs = 0) override;

 private:
  // Prefill: attention = attn_b + attn_k * sum_l2.
  //          other    = max(floor, other_b + other_k * n_tokens).
  double prefill_attn_b_;
  double prefill_attn_k_;
  double prefill_other_floor_;
  double prefill_other_b_;
  double prefill_other_k_;

  // Decode: attention = attn_b + attn_k * ctxsum.
  //         other    = max(floor, other_b + other_k * n_reqs).
  double decode_attn_b_;
  double decode_attn_k_;
  double decode_other_floor_;
  double decode_other_b_;
  double decode_other_k_;
};

// --- Unified roofline model ---

class UnifiedRooflineCostModel : public RolloutCostModel {
 public:
  UnifiedRooflineCostModel(double F, double W = 0, double G = 0,
                           double A_p = 0, double A_d = 0, int b = 16);

  double predict(const std::string& phase, double n_tokens, double sum_l2,
                 double ctxsum, int n_reqs, double prefill_ctx) override;

  std::tuple<double, double, double> decode_coeffs(int n_d,
                                                   double ctxsum) override;

  bool saturated(double n_tokens, double t2, double ctxsum,
                 int n_reqs = 0) override;

 private:
  double raw(const std::string& phase, double n_tokens, double sum_l2,
             double ctxsum) const;

  double F_;
  double W_;
  double G_;
  double A_p_;
  double A_d_;
  int b_;
};

}  // namespace srl::engine
