// dryrun/component/rollout/engine_cpp/src/kv_manager.cc
#include "engine/kv_manager.h"

#include <algorithm>
#include <type_traits>

namespace dryrun::engine {

namespace {

// Blocks needed to hold n_tokens, rounding up.
int blocks_for(int n_tokens, int block_size) {
  return (n_tokens + block_size - 1) / block_size;
}

}  // namespace

// --- Constructor ---

KvCacheManager::KvCacheManager(int num_blocks, int block_size,
                               bool enable_caching)
    : block_size_(block_size),
      enable_caching_(enable_caching),
      pool_(num_blocks, enable_caching) {
  if (block_size <= 0) {
    throw std::invalid_argument(
        "Expected block_size > 0, got " + std::to_string(block_size) + ".");
  }
}

// --- Read-only methods ---

int KvCacheManager::blocks_held(const Request& req) const {
  auto it = held_.find(req.rid);
  return (it != held_.end()) ? static_cast<int>(it->second.size()) : 0;
}

int KvCacheManager::peek_hit_tokens(const Request& req) const {
  if (!enable_caching_) return 0;
  int n_hit = 0;
  for (const auto& block_hash : block_hashes(req, block_size_)) {
    if (pool_.cached_block(block_hash) == nullptr) break;
    ++n_hit;
  }
  int tokens = n_hit * block_size_;
  if (n_hit > 0 && tokens >= req.prompt_len + req.generated) {
    tokens -= block_size_;
  }
  return tokens;
}

bool KvCacheManager::can_allocate(
    const Request& req, int n_tokens,
    const std::vector<KvBlock*>& hit_blocks) const {
  int needed = new_blocks_needed(req, n_tokens, hit_blocks);
  int revivable = 0;
  for (KvBlock* b : hit_blocks) {
    if (b->ref_cnt == 0) ++revivable;
  }
  return needed <= pool_.num_free_blocks() - revivable;
}

int KvCacheManager::new_blocks_needed(
    const Request& req, int n_tokens,
    const std::vector<KvBlock*>& hit_blocks) const {
  int computed = req.prefilled +
                 static_cast<int>(hit_blocks.size()) * block_size_;
  int required = blocks_for(computed + n_tokens, block_size_);
  return required - blocks_held(req) - static_cast<int>(hit_blocks.size());
}

HeldBlocksRestore KvCacheManager::capture_held(int rid) const {
  auto it = held_.find(rid);
  bool existed = (it != held_.end());
  std::vector<KvBlock*> old_blocks;
  if (existed) {
    old_blocks = it->second;  // copy
  }
  std::optional<int> old_n_cached;
  auto cit = n_cached_.find(rid);
  if (cit != n_cached_.end()) {
    old_n_cached = cit->second;
  }
  return HeldBlocksRestore{const_cast<KvCacheManager*>(this), rid, existed,
                           std::move(old_blocks), old_n_cached};
}

HeldBlocksTruncate KvCacheManager::capture_held_length(int rid) const {
  auto it = held_.find(rid);
  bool existed = (it != held_.end());
  size_t old_size = existed ? it->second.size() : 0;
  std::optional<int> old_n_cached;
  auto cit = n_cached_.find(rid);
  if (cit != n_cached_.end()) {
    old_n_cached = cit->second;
  }
  return HeldBlocksTruncate{const_cast<KvCacheManager*>(this), rid, existed,
                            old_size, old_n_cached};
}

// --- Template implementations ---

template <typename Recorder>
std::pair<std::vector<KvBlock*>, int>
KvCacheManager::computed_blocks_impl(const Request& req, Recorder& rec) {
  if (!enable_caching_) return {{}, 0};

  auto hashes = block_hashes(req, block_size_);
  std::vector<KvBlock*> hit;
  for (const auto& block_hash : hashes) {
    KvBlock* block = pool_.cached_block(block_hash);
    if (block == nullptr) break;
    hit.push_back(block);
  }

  // Capture diagnostic counters once per event for rollback.
  rec.record_once(
      JournalKey::of(this, JournalKey::kKvCounters),
      [this]() -> UndoEntry {
        return FieldSnapshot3{this, FieldSnapshot3::Kind::kCounters,
                              0.0, queries, false, hits};
      });
  queries += static_cast<int>(hashes.size());
  hits += static_cast<int>(hit.size());

  int tokens = static_cast<int>(hit.size()) * block_size_;
  // A full hit must still leave one block of work to do.
  if (!hit.empty() && tokens >= req.prompt_len + req.generated) {
    hit.pop_back();
    tokens -= block_size_;
    --hits;
  }
  return {std::move(hit), tokens};
}

template <typename Recorder>
std::optional<std::vector<KvBlock*>> KvCacheManager::allocate_impl(
    const Request& req, int n_tokens,
    const std::vector<KvBlock*>& hit_blocks, Recorder& rec) {
  if (!hit_blocks.empty() && !enable_caching_) {
    throw std::invalid_argument(
        "Hit blocks passed while prefix caching is disabled.");
  }
  if (!can_allocate(req, n_tokens, hit_blocks)) {
    return std::nullopt;
  }

  // The held list is only appended to below, so its length is enough to undo it.
  rec.record_once(
      JournalKey::of(this, JournalKey::kHeldBlocks, static_cast<uint32_t>(req.rid)),
      [this, rid = req.rid]() -> UndoEntry { return capture_held_length(rid); });

  pool_.touch(hit_blocks, rec);
  auto& held = held_[req.rid];
  for (KvBlock* b : hit_blocks) {
    held.push_back(b);
  }

  int needed = new_blocks_needed(req, n_tokens, hit_blocks);
  std::vector<KvBlock*> fresh;
  if (needed > 0) {
    fresh = pool_.new_blocks(needed, rec);
  }
  for (KvBlock* b : fresh) {
    held.push_back(b);
  }

  if (enable_caching_) {
    int computed = req.prefilled +
                   static_cast<int>(hit_blocks.size()) * block_size_;
    int n_full = (computed + n_tokens) / block_size_;
    auto cit = n_cached_.find(req.rid);
    int n_cached =
        (cit != n_cached_.end()) ? cit->second
                                 : static_cast<int>(hit_blocks.size());
    if (n_cached < n_full) {
      auto hashes = block_hashes(req, block_size_, n_cached, n_full);
      pool_.cache_full_blocks(held, hashes, n_cached, n_full, rec, n_cached);
    }
    n_cached_[req.rid] = n_full;
  }
  return fresh;
}

template <typename Recorder>
void KvCacheManager::free_impl(const Request& req, Recorder& rec) {
  // NOTE(claude): Journal the whole list on every free, not once per event. An
  // earlier allocation's entry holds only a length, so rollback (newest first)
  // needs this entry to restore the list before that one cuts it back.
  rec.record_lazy([this, rid = req.rid]() -> UndoEntry { return capture_held(rid); });

  auto it = held_.find(req.rid);
  if (it != held_.end()) {
    // Free in reverse order so the earliest blocks end up at the tail
    // of the free list (evicted last = better LRU for prompt blocks).
    std::vector<KvBlock*> rev(it->second.rbegin(), it->second.rend());
    held_.erase(it);
    pool_.free_blocks(rev, rec);
  }
  n_cached_.erase(req.rid);
}

template <typename Recorder>
void KvCacheManager::reset_hashes_impl(Recorder& rec) {
  if constexpr (std::is_same_v<Recorder, UndoLog>) {
    rec.record(CachedCountsRestore{this, n_cached_});
  }
  pool_.reset_hashes(rec);
  n_cached_.clear();
}

// --- Public overloads: UndoLog ---

std::pair<std::vector<KvBlock*>, int>
KvCacheManager::computed_blocks(const Request& req, UndoLog& log) {
  return computed_blocks_impl(req, log);
}

std::optional<std::vector<KvBlock*>> KvCacheManager::allocate(
    const Request& req, int n_tokens,
    const std::vector<KvBlock*>& hit_blocks, UndoLog& log) {
  return allocate_impl(req, n_tokens, hit_blocks, log);
}

void KvCacheManager::free(const Request& req, UndoLog& log) {
  free_impl(req, log);
}

void KvCacheManager::reset_hashes(UndoLog& log) {
  reset_hashes_impl(log);
}

// --- Public overloads: NullRecorder ---

std::pair<std::vector<KvBlock*>, int>
KvCacheManager::computed_blocks(const Request& req, NullRecorder& nr) {
  return computed_blocks_impl(req, nr);
}

std::optional<std::vector<KvBlock*>> KvCacheManager::allocate(
    const Request& req, int n_tokens,
    const std::vector<KvBlock*>& hit_blocks, NullRecorder& nr) {
  return allocate_impl(req, n_tokens, hit_blocks, nr);
}

void KvCacheManager::free(const Request& req, NullRecorder& nr) {
  free_impl(req, nr);
}

void KvCacheManager::reset_hashes(NullRecorder& nr) {
  reset_hashes_impl(nr);
}

}  // namespace dryrun::engine
