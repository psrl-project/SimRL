// dryrun/component/rollout/engine_cpp/include/engine/request.h
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace dryrun::engine {

enum class ReqStatus {
  kQueued,
  kWaiting,
  kRunning,
  kPreempted,
  kRecomputing,
  kDone,
  kConsumed,
  kDropped,
};

struct Segment {
  int version = 0;
  int tokens = 0;
};

struct Request {
  int rid;
  int prompt_len;
  int target_len;
  int v_traj;
  int group_id = 0;
  int group_index = 0;
  int source_group_id = 0;
  std::optional<int> instance_id;
  double admit_time = 0.0;
  double dispatch_time = 0.0;
  std::vector<Segment> segments;
  int generated = 0;
  int prefilled = 0;
  ReqStatus status = ReqStatus::kQueued;
  std::optional<double> complete_time;
  std::optional<int> consume_version;
  int num_aborts = 0;
  int num_preemptions = 0;
  int num_reroutes = 0;
  double router_enqueue_time = 0.0;
  int reprefill_tokens = 0;
  int prefix_hit_tokens = 0;

  int remaining() const { return target_len - generated; }
  int ctx() const { return prompt_len + generated; }

  int v_min() const {
    if (segments.empty()) return v_traj;
    return std::min_element(segments.begin(), segments.end(),
               [](const Segment& a, const Segment& b) {
                 return a.version < b.version;
               })->version;
  }

  int v_max() const {
    if (segments.empty()) return v_traj;
    return std::max_element(segments.begin(), segments.end(),
               [](const Segment& a, const Segment& b) {
                 return a.version < b.version;
               })->version;
  }

  void add_tokens(int version, int n) {
    if (n <= 0) return;
    if (segments.empty() || segments.back().version != version) {
      segments.push_back({version, 0});
    }
    segments.back().tokens += n;
    generated += n;
  }
};

using RequestPtr = std::shared_ptr<Request>;

}  // namespace dryrun::engine
