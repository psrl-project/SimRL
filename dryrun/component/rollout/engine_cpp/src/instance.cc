// dryrun/component/rollout/engine_cpp/src/instance.cc
#include "engine/instance.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace dryrun::engine {

// --- Constructor ---

NativeInstance::NativeInstance(
    std::shared_ptr<RolloutCostModel> cost, int token_budget, int kv_blocks,
    int block_size, int max_running, int max_concurrency, int chunk_cap,
    int instance_id, int tp, bool reject_if_kv_full, bool reject_if_waiting,
    bool reject_if_running_full, bool enable_prefix_caching)
    : cost_(std::move(cost)),
      M_(token_budget),
      kv_blocks_(kv_blocks),
      block_size_(block_size),
      max_running_(max_running),
      max_concurrency_(max_concurrency),
      chunk_cap_(chunk_cap),
      instance_id_(instance_id),
      tp_(tp),
      reject_if_kv_full_(reject_if_kv_full),
      reject_if_waiting_(reject_if_waiting),
      reject_if_running_full_(reject_if_running_full),
      enable_prefix_caching_(enable_prefix_caching),
      kv_(kv_blocks, block_size, enable_prefix_caching) {}

// --- Public accessors ---

InstanceLoad NativeInstance::get_load() const {
  int total_ctx = 0;
  for (const auto& r : running_) {
    total_ctx += r->ctx();
  }
  return {
      instance_id_,
      static_cast<int>(waiting_.size()),
      static_cast<int>(running_.size()),
      static_cast<double>(blocks_used()) / std::max(1, kv_blocks_),
      total_ctx,
      tp_,
  };
}

int NativeInstance::committed_blocks() const {
  int committed = 0;
  for (const auto& r : waiting_) committed += r->ctx();
  for (const auto& r : running_) committed += r->ctx();
  return blocks_for(committed, block_size_);
}

double NativeInstance::step_time_delta(const RequestPtr& req) const {
  auto [n_d, ctxsum] = decode_batch();
  double before =
      (n_d > 0) ? cost_->step_time(n_d, 0, ctxsum, n_d) : 0.0;
  double after = cost_->step_time(n_d + 1, 0, ctxsum + req->ctx(), n_d + 1);
  return std::max(0.0, after - before);
}

// --- Snapshot / undo ---

void NativeInstance::save_snapshot() {
  if (spare_log_) {
    log_ = std::move(spare_log_);
  } else {
    log_ = std::make_unique<UndoLog>();
  }

  // Record instance scalar fields.
  log_->record(FieldSnapshot3{
      this, FieldSnapshot3::Kind::kInstanceScalars,
      t_local_, n_preemptions_, last_decode_blocked_, 0});

  // Record waiting and running list copies.
  log_->record(VectorRestore{
      this, VectorRestore::Which::kWaiting,
      std::vector<RequestPtr>(waiting_.begin(), waiting_.end())});
  log_->record(VectorRestore{
      this, VectorRestore::Which::kRunning,
      std::vector<RequestPtr>(running_.begin(), running_.end())});

  // Record completed and step_stats lengths.
  log_->record(VectorTruncate{
      this, VectorTruncate::Which::kCompleted, completed_.size()});
  log_->record(VectorTruncate{
      this, VectorTruncate::Which::kStepStats, step_stats_.size()});

  // Record per-request snapshots.
  for (const auto& req : waiting_) {
    log_->record_once(JournalKey::of(req.get(), JournalKey::kRequestFields),
                      [&]() -> UndoEntry { return RequestSnapshot::capture(req); });
  }
  for (const auto& req : running_) {
    log_->record_once(JournalKey::of(req.get(), JournalKey::kRequestFields),
                      [&]() -> UndoEntry { return RequestSnapshot::capture(req); });
  }
}

void NativeInstance::commit_last_event() {
  recycle_log();
}

void NativeInstance::undo_last_event() {
  if (!log_) {
    throw std::runtime_error("No snapshot to undo.");
  }
  log_->rollback();
  recycle_log();
}

// --- Internal helpers ---

int NativeInstance::n_in_batch() const {
  int count = static_cast<int>(running_.size());
  for (const auto& r : waiting_) {
    if (r->prefilled > 0) ++count;
  }
  return count;
}

bool NativeInstance::can_run_more() const {
  return max_concurrency_ < 0 ||
         static_cast<int>(running_.size()) < max_concurrency_;
}

bool NativeInstance::can_schedule_new() const {
  return max_concurrency_ < 0 || n_in_batch() < max_concurrency_;
}

void NativeInstance::promote_ready() {
  auto it = waiting_.begin();
  while (it != waiting_.end()) {
    if (!can_run_more()) break;
    const auto& req = *it;
    if (req->prompt_len + req->generated - req->prefilled <= 0) {
      req->status = ReqStatus::kRunning;
      running_.push_back(req);
      it = waiting_.erase(it);
    } else {
      ++it;
    }
  }
}

int NativeInstance::apply_prefix_hit(const RequestPtr& req) {
  if (!enable_prefix_caching_ || req->prefilled > 0) return 0;

  auto [hit_blocks, hit_tokens] =
      log_ ? kv_.computed_blocks(*req, *log_)
           : kv_.computed_blocks(*req, kNullRecorder);
  if (hit_tokens <= 0) return 0;
  if (!kv_.can_allocate(*req, 1, hit_blocks)) return 0;

  if (log_) {
    kv_.allocate(*req, 0, hit_blocks, *log_);
  } else {
    kv_.allocate(*req, 0, hit_blocks, kNullRecorder);
  }

  req->prefilled = hit_tokens;
  req->prefix_hit_tokens += hit_tokens;
  return hit_tokens;
}

std::pair<std::vector<RequestPtr>, bool> NativeInstance::decode_running_once(
    int version) {
  int n_decode = std::min(static_cast<int>(running_.size()), M_);
  std::vector<RequestPtr> stepped;
  bool blocked = false;

  // Find oldest request for the decode-first guarantee.
  RequestPtr oldest;
  if (!running_.empty()) {
    oldest = *std::min_element(
        running_.begin(), running_.end(),
        [](const RequestPtr& a, const RequestPtr& b) {
          return std::tie(a->dispatch_time, a->rid) <
                 std::tie(b->dispatch_time, b->rid);
        });
  }

  int free = free_blocks();
  for (int i = 0; i < n_decode; ++i) {
    const auto& r = running_[i];
    bool grows = (r->prefilled % block_size_) == 0;
    if (grows && free <= 0 && r != oldest) {
      blocked = true;
      continue;
    }

    std::vector<KvBlock*> empty_hits;
    auto granted = log_ ? kv_.allocate(*r, 1, empty_hits, *log_)
                        : kv_.allocate(*r, 1, empty_hits, kNullRecorder);
    if (!granted.has_value()) {
      blocked = true;
      continue;
    }
    r->add_tokens(version, 1);
    r->prefilled += 1;
    if (grows) --free;
    stepped.push_back(r);
  }
  return {std::move(stepped), blocked};
}

int NativeInstance::prefill_chunk(const Request& req, int prefilled, int budget,
                                 int free) const {
  int chunk = std::min(req.prompt_len + req.generated - prefilled, budget);
  if (chunk_cap_ > 0) {
    chunk = std::min(chunk, chunk_cap_);
  }
  int have = blocks_for(prefilled, block_size_);
  chunk = std::min(chunk, (have + std::max(0, free)) * block_size_ - prefilled);
  // Pool is the authority on capacity.
  std::vector<KvBlock*> empty_hits;
  while (chunk > 0 && !kv_.can_allocate(req, chunk, empty_hits)) {
    chunk -= block_size_;
  }
  return std::max(0, chunk);
}

std::pair<double, std::vector<RequestPtr>> NativeInstance::engine_step(
    int version) {
  int n_d = static_cast<int>(running_.size());
  int n_decode = std::min(n_d, M_);
  int budget = M_ - n_decode;
  double t2 = 0;

  double prefill_ctx_sum = 0;
  int prefill_tokens = 0;
  int n_prefill_reqs = 0;
  int admitted_blocks = 0;
  std::vector<RequestPtr> finished_prefill;
  int batch_count = n_in_batch();

  auto [stepped, decode_blocked] = decode_running_once(version);
  std::vector<RequestPtr> newly_done;
  for (const auto& r : stepped) {
    if (r->remaining() <= 0) newly_done.push_back(r);
  }

  int free = free_blocks();
  int prefill_hit = 0;

  // Iterate over the waiting list. Copy the list to iterate safely since
  // requests may be moved during the loop.
  std::vector<RequestPtr> waiting_copy(waiting_.begin(), waiting_.end());
  for (const auto& req : waiting_copy) {
    if (budget <= 0) break;

    if (req->prefilled == 0) {
      if (decode_blocked) break;
      if (max_concurrency_ >= 0 && batch_count >= max_concurrency_) break;

      if (!running_.empty() || admitted_blocks > 0) {
        int want = req->prompt_len + req->generated;
        if (blocks_for(want, block_size_) > free_blocks() - admitted_blocks) {
          break;
        }
        admitted_blocks += blocks_for(want, block_size_);
      }
      batch_count += 1;
      prefill_hit += apply_prefix_hit(req);
      free = free_blocks();
    }

    int need = req->prompt_len + req->generated - req->prefilled;
    if (need <= 0) {
      finished_prefill.push_back(req);
      continue;
    }
    int have = blocks_for(req->prefilled, block_size_);
    int chunk = prefill_chunk(*req, req->prefilled, budget, free);
    if (chunk <= 0) break;

    std::vector<KvBlock*> empty_hits;
    auto granted = log_ ? kv_.allocate(*req, chunk, empty_hits, *log_)
                        : kv_.allocate(*req, chunk, empty_hits, kNullRecorder);
    if (!granted.has_value()) {
      throw std::runtime_error(
          "KV pool refused prefill tokens after can_allocate approved them.");
    }
    free -= blocks_for(req->prefilled + chunk, block_size_) - have;
    double extent = static_cast<double>(req->prefilled + chunk);
    prefill_ctx_sum += extent;
    req->prefilled += chunk;
    prefill_tokens += chunk;
    n_prefill_reqs += 1;
    t2 += extent * extent;
    budget -= chunk;
    if (req->prefilled >= req->prompt_len + req->generated) {
      finished_prefill.push_back(req);
    }
  }

  double ctxsum = 0;
  for (const auto& r : running_) ctxsum += r->ctx();
  int n_tokens = n_decode + prefill_tokens;
  int n_reqs = n_d + n_prefill_reqs;
  double dt = cost_->step_time(n_tokens, t2, ctxsum, n_reqs, prefill_ctx_sum);
  if (dt <= 0) {
    throw std::runtime_error("Cost model returned non-positive step time.");
  }

  step_stats_.emplace_back(
      n_tokens, n_decode, prefill_tokens,
      cost_->saturated(n_tokens, t2, ctxsum, n_reqs),
      /*steps=*/1, /*saturated_steps=*/0, dt, "mixed",
      /*t=*/t_local_ + dt, prefill_hit, prefill_tokens);

  for (const auto& req : finished_prefill) {
    if (!can_run_more()) break;
    auto it = std::find(waiting_.begin(), waiting_.end(), req);
    if (it != waiting_.end()) {
      waiting_.erase(it);
      req->status = ReqStatus::kRunning;
      running_.push_back(req);
    }
  }
  last_decode_blocked_ = decode_blocked;
  return {dt, std::move(newly_done)};
}

std::pair<int, double> NativeInstance::decode_batch() const {
  int n_d = static_cast<int>(running_.size());
  double ctxsum = 0;
  for (const auto& r : running_) ctxsum += r->ctx();
  return {n_d, ctxsum};
}

std::optional<int> NativeInstance::kv_exhaustion_steps() const {
  int B = block_size_;
  std::vector<int> residues(B, 0);
  for (const auto& r : running_) {
    residues[r->prefilled % B] += 1;
  }
  int base = blocks_used();
  if (base > kv_blocks_) return 0;

  int cap = 0;
  for (const auto& r : running_) {
    cap = std::max(cap, r->remaining());
  }
  if (cap == 0) return std::nullopt;

  int n_run = static_cast<int>(running_.size());
  int target = kv_blocks_ + 1;
  int best = cap + 1;
  int blocks_s = base;
  for (int s = 0; s < B; ++s) {
    int need = target - blocks_s;
    int steps;
    if (need <= 0) {
      steps = s;
    } else {
      // ceil(need / n_run) * B + s.
      steps = ((need + n_run - 1) / n_run) * B + s;
    }
    if (steps < best) best = steps;
    blocks_s += residues[(B - s) % B];
  }
  return (best <= cap) ? std::optional<int>(best) : std::nullopt;
}

double NativeInstance::step_time_estimate() const {
  int n_d = static_cast<int>(running_.size());
  int n_decode = std::min(n_d, M_);
  int budget = M_ - n_decode;
  double t2 = 0;
  double prefill_ctx_sum = 0;
  int prefill_tokens = 0;
  int free = free_blocks();
  // Replay the decode pass's block demand, so the gates below see the pool and the
  // blocked flag of this step rather than the previous one.
  bool decode_blocked = false;
  for (int i = 0; i < n_decode; ++i) {
    if (running_[i]->prefilled % block_size_ == 0) {
      if (free <= 0) {
        decode_blocked = true;
      } else {
        --free;
      }
    }
  }
  std::unordered_map<uintptr_t, int> effective;
  int n_prefill_reqs = 0;
  int admitted_blocks = 0;
  int batch_count = n_in_batch();

  for (const auto& req : waiting_) {
    if (budget <= 0) break;

    if (req->prefilled == 0) {
      if (decode_blocked) break;
      if (max_concurrency_ >= 0 && batch_count >= max_concurrency_) break;

      if (!running_.empty() || admitted_blocks > 0) {
        int want = req->prompt_len + req->generated;
        if (blocks_for(want, block_size_) > free - admitted_blocks) {
          break;
        }
        admitted_blocks += blocks_for(want, block_size_);
      }
      batch_count += 1;
    }

    uintptr_t key = reinterpret_cast<uintptr_t>(req.get());
    auto it = effective.find(key);
    int eff = (it != effective.end()) ? it->second : req->prefilled;
    if (eff == 0) {
      eff = kv_.peek_hit_tokens(*req);
      effective[key] = eff;
      free -= blocks_for(eff, block_size_);
    }
    int need = req->prompt_len + req->generated - eff;
    if (need <= 0) continue;

    int have = blocks_for(eff, block_size_);
    int chunk = prefill_chunk(*req, eff, budget, free);
    if (chunk <= 0) break;

    free -= blocks_for(eff + chunk, block_size_) - have;
    effective[key] = eff + chunk;
    double extent = static_cast<double>(eff + chunk);
    prefill_ctx_sum += extent;
    prefill_tokens += chunk;
    n_prefill_reqs += 1;
    t2 += extent * extent;
    budget -= chunk;
  }

  double ctxsum = 0;
  for (const auto& r : running_) ctxsum += r->ctx();
  int n_tokens = n_decode + prefill_tokens;
  return cost_->step_time(n_tokens, t2, ctxsum, n_d + n_prefill_reqs,
                          prefill_ctx_sum);
}

EngineEvent NativeInstance::iteration_event(int version) {
  auto [dt, newly] = engine_step(version);
  t_local_ += dt;
  if (!newly.empty()) {
    auto done = retire(newly);
    std::vector<RequestPtr> evicted;
    if (deadlocked()) {
      evicted = preempt_for_progress();
    }
    return {EventKind::kRequestComplete, instance_id_, t_local_,
            std::move(done), std::move(evicted)};
  }
  if (deadlocked()) {
    return {EventKind::kPreempt, instance_id_, t_local_, {},
            preempt_for_progress()};
  }
  return {EventKind::kIterationDone, instance_id_, t_local_, {}, {}};
}

void NativeInstance::decode_segment(int version, int k) {
  std::vector<KvBlock*> empty_hits;
  for (const auto& r : running_) {
    auto granted = log_ ? kv_.allocate(*r, k, empty_hits, *log_)
                        : kv_.allocate(*r, k, empty_hits, kNullRecorder);
    if (!granted.has_value()) {
      throw std::runtime_error(
          "KV pool refused decode tokens for a request. The step count must "
          "be one the pool can satisfy for every request.");
    }
    r->add_tokens(version, k);
    r->prefilled += k;
  }
}

EngineEvent NativeInstance::decode_segment_event(int version) {
  auto [n_d, ctxsum] = decode_batch();
  std::vector<int> remaining_vec;
  remaining_vec.reserve(running_.size());
  for (const auto& r : running_) remaining_vec.push_back(r->remaining());
  auto [k_done, _] = next_completion(remaining_vec);
  auto k_kv = kv_exhaustion_steps();

  bool kv_bound = k_kv.has_value() && *k_kv <= k_done;
  int k = kv_bound ? std::max(0, *k_kv - 1) : k_done;

  if (k > 0) {
    decode_segment(version, k);
    double dt = cost_->decode_segment_time(n_d, ctxsum, k);
    t_local_ += dt;
    int saturated_steps = cost_->decode_saturated_steps(n_d, ctxsum, k);
    step_stats_.emplace_back(
        n_d * k, n_d * k, 0,
        saturated_steps > 0,
        k, saturated_steps, dt, "decode", t_local_);
  }

  if (kv_bound) {
    return {EventKind::kPreempt, instance_id_, t_local_, {},
            preempt_until_decodable()};
  }

  std::vector<RequestPtr> newly;
  for (const auto& r : running_) {
    if (r->remaining() <= 0) newly.push_back(r);
  }
  auto done = retire(newly);
  return {EventKind::kRequestComplete, instance_id_, t_local_,
          std::move(done), {}};
}

// --- Public interface ---

std::optional<EngineEvent> NativeInstance::advance_one_event(int version) {
  promote_ready();
  if (running_.empty() && waiting_.empty()) {
    recycle_log();
    return std::nullopt;
  }
  save_snapshot();
  if (!waiting_.empty()) {
    return iteration_event(version);
  }
  return decode_segment_event(version);
}

std::optional<EngineEvent> NativeInstance::advance_one_event_committed(
    int version) {
  recycle_log();
  promote_ready();
  if (running_.empty() && waiting_.empty()) {
    return std::nullopt;
  }
  if (!waiting_.empty()) {
    return iteration_event(version);
  }
  return decode_segment_event(version);
}

NativeInstance::IterationBatch NativeInstance::advance_iterations(
    int version, bool first_speculative, bool iter_speculative, bool has_top,
    double t_top, int iid_top, double deadline) {
  IterationBatch out;
  out.last_committed_t = t_local_;
  bool speculative = first_speculative;
  while (true) {
    out.n_stats_committed = step_stats_.size();
    std::optional<EngineEvent> ev = speculative
                                        ? advance_one_event(version)
                                        : advance_one_event_committed(version);
    if (!ev.has_value()) return out;
    bool earliest = !has_top || ev->t < t_top ||
                    (ev->t == t_top && instance_id_ < iid_top);
    if (ev->kind != EventKind::kIterationDone || ev->t > deadline || !earliest) {
      out.pending = std::move(ev);
      return out;
    }
    commit_last_event();
    out.n_committed += 1;
    out.last_committed_t = ev->t;
    speculative = iter_speculative;
  }
}

bool NativeInstance::add_request(const RequestPtr& req, double t) {
  int need = blocks_for(req->prompt_len + req->target_len, block_size_);
  if (need > kv_blocks_) {
    throw std::invalid_argument(
        "Request " + std::to_string(req->rid) + " needs " +
        std::to_string(need) + " KV blocks but instance " +
        std::to_string(instance_id_) + " only has " +
        std::to_string(kv_blocks_) + " blocks. So it can never be admitted.");
  }
  if (max_running_ >= 0 && n_active() >= max_running_) return false;
  if (reject_if_waiting_ && !waiting_.empty()) return false;
  if (reject_if_running_full_ && !can_schedule_new()) return false;
  if (reject_if_kv_full_) {
    int committed = 0;
    for (const auto& r : waiting_) committed += r->ctx();
    for (const auto& r : running_) committed += r->ctx();
    if (blocks_for(committed + req->prompt_len, block_size_) > kv_blocks_) {
      return false;
    }
  }
  req->status = ReqStatus::kWaiting;
  req->dispatch_time = t;
  req->instance_id = instance_id_;
  waiting_.push_back(req);
  return true;
}

std::vector<RequestPtr> NativeInstance::run_until(double t_limit, int version) {
  std::vector<RequestPtr> done;
  while (t_local_ < t_limit) {
    auto [got, reached] = advance_to(t_limit, version);
    done.insert(done.end(), got.begin(), got.end());
    if (reached) break;
    if (running_.empty() && waiting_.empty()) break;
  }
  return done;
}

std::tuple<std::vector<RequestPtr>, bool> NativeInstance::advance_to(
    double t_limit, int version, int max_steps) {
  std::vector<RequestPtr> done;
  int steps = 0;
  while (t_local_ < t_limit) {
    promote_ready();
    if (!waiting_.empty()) {
      if (steps >= max_steps) return {std::move(done), false};
      ++steps;
      double dt = step_time_estimate();
      if (t_local_ + dt > t_limit) {
        t_local_ = t_limit;
        return {std::move(done), true};
      }
      auto [step_dt, newly] = engine_step(version);
      t_local_ += dt;
      auto retired = retire(newly);
      done.insert(done.end(), retired.begin(), retired.end());
      if (!newly.empty()) return {std::move(done), false};
      if (deadlocked()) {
        if (preempt_for_progress().empty()) {
          t_local_ = t_limit;
          return {std::move(done), true};
        }
      }
      continue;
    }

    if (running_.empty()) {
      t_local_ = t_limit;
      return {std::move(done), true};
    }

    auto [n_d, ctxsum] = decode_batch();
    std::vector<int> remaining_vec;
    remaining_vec.reserve(running_.size());
    for (const auto& r : running_) remaining_vec.push_back(r->remaining());
    auto [k_done, _] = next_completion(remaining_vec);
    auto k_kv = kv_exhaustion_steps();
    bool kv_bound = k_kv.has_value() && *k_kv <= k_done;
    int k_int = kv_bound ? std::max(0, *k_kv - 1) : k_done;
    double budget_time = t_limit - t_local_;
    int k_budget = cost_->decode_steps_within(n_d, ctxsum, budget_time);

    int k = std::min(k_int, k_budget);
    if (k > 0) {
      decode_segment(version, k);
      double dt = cost_->decode_segment_time(n_d, ctxsum, k);
      t_local_ += dt;
      int saturated_steps = cost_->decode_saturated_steps(n_d, ctxsum, k);
      step_stats_.emplace_back(
          n_d * k, n_d * k, 0,
          saturated_steps > 0,
          k, saturated_steps, dt, "decode", t_local_);
    }

    if (blocks_used() > kv_blocks_) {
      preempt_until_decodable();
      return {std::move(done), false};
    }

    if (k == k_budget && k < k_int) {
      t_local_ = t_limit;
      return {std::move(done), true};
    }

    std::vector<RequestPtr> newly;
    for (const auto& r : running_) {
      if (r->remaining() <= 0) newly.push_back(r);
    }
    if (!newly.empty()) {
      auto retired = retire(newly);
      done.insert(done.end(), retired.begin(), retired.end());
      return {std::move(done), false};
    }
    if (kv_bound && k >= k_int) {
      preempt_until_decodable();
      return {std::move(done), false};
    }
    if (k == 0) {
      t_local_ = t_limit;
      return {std::move(done), true};
    }
  }
  return {std::move(done), true};
}

// --- Deadlock and preemption ---

bool NativeInstance::deadlocked() const {
  if (running_.empty() || free_blocks() > 0) return false;
  bool prefill_blocked = false;
  for (const auto& r : waiting_) {
    if (r->prompt_len + r->generated - r->prefilled > 0) {
      prefill_blocked = true;
      break;
    }
  }
  return last_decode_blocked_ &&
         (prefill_blocked || static_cast<int>(running_.size()) > 1);
}

std::vector<RequestPtr> NativeInstance::preempt_for_progress() {
  return preempt_until_decodable();
}

std::vector<RequestPtr> NativeInstance::retire(
    const std::vector<RequestPtr>& reqs) {
  std::vector<RequestPtr> out;
  for (const auto& r : reqs) {
    auto it = std::find(running_.begin(), running_.end(), r);
    if (it != running_.end()) {
      running_.erase(it);
    }
    if (log_) {
      kv_.free(*r, *log_);
    } else {
      kv_.free(*r, kNullRecorder);
    }
    r->status = ReqStatus::kDone;
    r->complete_time = t_local_;
    completed_.push_back(r);
    out.push_back(r);
  }
  return out;
}

RequestPtr NativeInstance::preempt_one(RequestPtr victim) {
  if (running_.empty()) return nullptr;
  if (!victim) {
    victim = running_.back();
  }
  auto it = std::find(running_.begin(), running_.end(), victim);
  if (it != running_.end()) {
    running_.erase(it);
  }
  victim->status = ReqStatus::kPreempted;
  victim->num_preemptions += 1;
  victim->reprefill_tokens += victim->ctx();
  if (log_) {
    kv_.free(*victim, *log_);
  } else {
    kv_.free(*victim, kNullRecorder);
  }
  victim->prefilled = 0;
  waiting_.insert(waiting_.begin(), victim);
  n_preemptions_ += 1;
  return victim;
}

std::vector<RequestPtr> NativeInstance::preempt_until_decodable() {
  std::vector<RequestPtr> evicted;
  while (!running_.empty()) {
    int needed = 0;
    for (const auto& r : running_) {
      if ((r->prefilled % block_size_) == 0) ++needed;
    }
    if (free_blocks() >= needed) break;

    // Find oldest across both lists.
    RequestPtr oldest;
    for (const auto& r : running_) {
      if (!oldest ||
          std::tie(r->dispatch_time, r->rid) <
              std::tie(oldest->dispatch_time, oldest->rid)) {
        oldest = r;
      }
    }
    for (const auto& r : waiting_) {
      if (!oldest ||
          std::tie(r->dispatch_time, r->rid) <
              std::tie(oldest->dispatch_time, oldest->rid)) {
        oldest = r;
      }
    }

    std::vector<RequestPtr> victims;
    for (const auto& r : running_) {
      if (r != oldest) victims.push_back(r);
    }
    if (victims.empty()) break;

    auto victim = preempt_one(pick_victim(victims));
    if (!victim) break;
    evicted.push_back(victim);
  }
  return evicted;
}

RequestPtr NativeInstance::pick_victim(
    const std::vector<RequestPtr>& victims) const {
  if (eviction_) {
    RequestPtr chosen = eviction_(victims, t_local_);
    if (chosen) return chosen;
  }
  // The default is newest-dispatched first, vLLM's reverse arrival order.
  return *std::max_element(
      victims.begin(), victims.end(),
      [](const RequestPtr& a, const RequestPtr& b) {
        return std::tie(a->dispatch_time, a->rid) <
               std::tie(b->dispatch_time, b->rid);
      });
}

void NativeInstance::release(const RequestPtr& req, bool keep_tokens) {
  req->num_aborts += 1;
  if (keep_tokens) {
    req->reprefill_tokens += req->ctx();
  }
  if (log_) {
    kv_.free(*req, *log_);
  } else {
    kv_.free(*req, kNullRecorder);
  }
  req->prefilled = 0;
}

std::vector<RequestPtr> NativeInstance::abort_all(bool requeue) {
  std::vector<RequestPtr> aborted;
  aborted.reserve(running_.size() + waiting_.size());
  for (const auto& r : running_) aborted.push_back(r);
  for (const auto& r : waiting_) aborted.push_back(r);
  for (const auto& r : aborted) {
    release(r, /*keep_tokens=*/true);
    r->status = ReqStatus::kWaiting;
  }
  if (requeue) {
    waiting_.assign(aborted.begin(), aborted.end());
  } else {
    waiting_.clear();
  }
  running_.clear();
  if (log_) {
    kv_.reset_hashes(*log_);
  } else {
    kv_.reset_hashes(kNullRecorder);
  }
  return aborted;
}

std::vector<RequestPtr> NativeInstance::cancel(
    const std::vector<RequestPtr>& reqs, bool keep_tokens, bool requeue) {
  std::vector<RequestPtr> canceled;
  for (const auto& r : reqs) {
    auto it_r = std::find(running_.begin(), running_.end(), r);
    if (it_r != running_.end()) {
      running_.erase(it_r);
    } else {
      auto it_w = std::find(waiting_.begin(), waiting_.end(), r);
      if (it_w != waiting_.end()) {
        waiting_.erase(it_w);
      } else {
        continue;
      }
    }
    release(r, keep_tokens);
    if (keep_tokens) {
      r->status = ReqStatus::kWaiting;
      if (requeue) {
        waiting_.push_back(r);
      }
    }
    canceled.push_back(r);
  }
  return canceled;
}

}  // namespace dryrun::engine
