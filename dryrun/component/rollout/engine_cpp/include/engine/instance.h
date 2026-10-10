// dryrun/component/rollout/engine_cpp/include/engine/instance.h
//
// A single simulated engine instance with continuous batching, KV cache
// accounting, and preemption.
//
// Port of dryrun/component/rollout/engine/instance.py.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "engine/cost_model.h"
#include "engine/event.h"
#include "engine/kv_manager.h"
#include "engine/request.h"
#include "engine/undo.h"

namespace dryrun::engine {

// Ceiling division for block counts.
inline int blocks_for(int n_tokens, int block_size) {
  return (n_tokens + block_size - 1) / block_size;
}

struct StepStat {
  int n_tokens = 0;
  int n_decode = 0;
  int n_prefill = 0;
  bool saturated = false;
  int steps = 1;
  int saturated_steps = 0;
  double dt = 0.0;
  std::string phase = "decode";
  double t = 0.0;
  int n_prefill_hit = 0;
  int n_prefill_computed = 0;

  StepStat() = default;
  StepStat(int n_tokens, int n_decode, int n_prefill, bool saturated,
           int steps = 1, int saturated_steps = 0, double dt = 0.0,
           const std::string& phase = "decode", double t = 0.0,
           int n_prefill_hit = 0, int n_prefill_computed = 0)
      : n_tokens(n_tokens),
        n_decode(n_decode),
        n_prefill(n_prefill),
        saturated(saturated),
        steps(steps),
        saturated_steps(saturated_steps),
        dt(dt),
        phase(phase),
        t(t),
        n_prefill_hit(n_prefill_hit),
        n_prefill_computed(n_prefill_computed) {
    if (this->saturated && this->saturated_steps == 0) {
      this->saturated_steps = this->steps;
    }
  }
};

struct InstanceLoad {
  int instance_id = 0;
  int n_waiting = 0;
  int n_running = 0;
  double kv_utilization = 0.0;
  int total_ctx = 0;
  int tp = 1;

  int n_active() const { return n_waiting + n_running; }
};

class NativeInstance {
 public:
  NativeInstance(std::shared_ptr<RolloutCostModel> cost, int token_budget,
                int kv_blocks, int block_size = 16,
                int max_running = -1,
                int max_concurrency = -1,
                int chunk_cap = -1,
                int instance_id = 0, int tp = 1,
                bool reject_if_kv_full = false,
                bool reject_if_waiting = false,
                bool reject_if_running_full = false,
                bool enable_prefix_caching = true);

  // --- Public accessors ---

  int n_active() const {
    return static_cast<int>(waiting_.size() + running_.size());
  }
  InstanceLoad get_load() const;
  int blocks_used() const { return kv_.num_used_blocks(); }
  int free_blocks() const { return kv_.num_free_blocks(); }
  int committed_blocks() const;
  double step_time_delta(const RequestPtr& req) const;

  // Expose the Python-side attribute names.
  double t_local() const { return t_local_; }
  int instance_id() const { return instance_id_; }
  int tp() const { return tp_; }
  int n_preemptions() const { return n_preemptions_; }
  int kv_queries() const { return kv_.queries; }
  int kv_hits() const { return kv_.hits; }

  const std::vector<RequestPtr>& waiting() const { return waiting_; }
  const std::vector<RequestPtr>& running() const { return running_; }
  const std::vector<RequestPtr>& completed() const { return completed_; }
  const std::vector<StepStat>& step_stats() const { return step_stats_; }

  // --- Mutation interface ---

  bool add_request(const RequestPtr& req, double t);

  // Returns the event, or nullopt if idle.
  std::optional<EngineEvent> advance_one_event(int version);

  void commit_last_event();
  void undo_last_event();

  std::vector<RequestPtr> abort_all(bool requeue = true);
  std::vector<RequestPtr> cancel(const std::vector<RequestPtr>& reqs,
                                 bool keep_tokens = true,
                                 bool requeue = true);

  std::tuple<std::vector<RequestPtr>, bool> advance_to(double t_limit,
                                                        int version,
                                                        int max_steps = 100000);

  std::vector<RequestPtr> run_until(double t_limit, int version);

 private:
  friend struct FieldSnapshot3;
  friend struct VectorRestore;
  friend struct VectorTruncate;

  // --- Internal engine methods ---

  int n_in_batch() const;
  bool can_run_more() const;
  bool can_schedule_new() const;
  void promote_ready();
  std::pair<std::vector<RequestPtr>, bool> decode_running_once(int version);
  int prefill_chunk(const Request& req, int prefilled, int budget,
                    int free) const;
  int apply_prefix_hit(const RequestPtr& req);
  std::pair<double, std::vector<RequestPtr>> engine_step(int version);
  std::pair<int, double> decode_batch() const;
  std::optional<int> kv_exhaustion_steps() const;
  double step_time_estimate() const;
  EngineEvent iteration_event(int version);
  void decode_segment(int version, int k);
  EngineEvent decode_segment_event(int version);
  bool deadlocked() const;
  std::vector<RequestPtr> preempt_for_progress();
  std::vector<RequestPtr> retire(const std::vector<RequestPtr>& reqs);
  RequestPtr preempt_one(RequestPtr victim = nullptr);
  std::vector<RequestPtr> preempt_until_decodable();
  void release(const RequestPtr& req, bool keep_tokens);
  void save_snapshot();

  // --- Fields ---

  std::shared_ptr<RolloutCostModel> cost_;
  int M_;  // token_budget
  int kv_blocks_;
  int block_size_;
  int max_running_;      // -1 means no limit
  int max_concurrency_;  // -1 means no limit
  int chunk_cap_;        // -1 means no limit
  int instance_id_;
  int tp_;
  bool reject_if_kv_full_;
  bool reject_if_waiting_;
  bool reject_if_running_full_;
  bool enable_prefix_caching_;

  mutable KvCacheManager kv_;
  double t_local_ = 0.0;
  std::vector<RequestPtr> waiting_;
  std::vector<RequestPtr> running_;
  std::vector<StepStat> step_stats_;
  int n_preemptions_ = 0;
  std::vector<RequestPtr> completed_;
  bool last_decode_blocked_ = false;

  // Undo log for speculative events. nullptr when no event is in flight.
  std::unique_ptr<UndoLog> log_;
};

}  // namespace dryrun::engine
