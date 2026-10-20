// srl/component/rollout/engine_cpp/include/engine/kv_manager.h
//
// Per-instance KV cache manager: the engine's view of the block pool.
//
// Port of srl/component/rollout/engine/kv/manager.py. NativeInstance talks
// to this and never to BlockPool directly, so occupancy accounting and hit
// lookup stay in one place.
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "engine/block.h"
#include "engine/lineage.h"
#include "engine/pool.h"
#include "engine/request.h"
#include "engine/undo.h"

namespace srl::engine {

class KvCacheManager {
 public:
  // Args:
  //   num_blocks: Pool capacity.
  //   block_size: Tokens per block.
  //   enable_caching: When false no hash is recorded, so nothing hits.
  KvCacheManager(int num_blocks, int block_size = 16,
                 bool enable_caching = true);

  KvCacheManager(const KvCacheManager&) = delete;
  KvCacheManager& operator=(const KvCacheManager&) = delete;
  KvCacheManager(KvCacheManager&&) = delete;
  KvCacheManager& operator=(KvCacheManager&&) = delete;

  int block_size() const { return block_size_; }
  bool enable_caching() const { return enable_caching_; }
  int num_used_blocks() const { return pool_.num_used_blocks(); }
  int num_free_blocks() const { return pool_.num_free_blocks(); }
  double usage() const { return pool_.usage(); }

  // How many blocks this request currently holds.
  int blocks_held(const Request& req) const;

  // Access the underlying pool (for free_list_ids, etc.).
  BlockPool& pool() { return pool_; }
  const BlockPool& pool() const { return pool_; }

  // --- Recorded overloads (speculative / undo-tracked paths) ---

  std::pair<std::vector<KvBlock*>, int> computed_blocks(const Request& req,
                                                        UndoLog& log);
  std::optional<std::vector<KvBlock*>> allocate(const Request& req,
                                                int n_tokens,
                                                const std::vector<KvBlock*>& hit_blocks,
                                                UndoLog& log);
  void free(const Request& req, UndoLog& log);
  void reset_hashes(UndoLog& log);

  // --- NullRecorder overloads (committed paths) ---

  std::pair<std::vector<KvBlock*>, int> computed_blocks(const Request& req,
                                                        NullRecorder&);
  std::optional<std::vector<KvBlock*>> allocate(const Request& req,
                                                int n_tokens,
                                                const std::vector<KvBlock*>& hit_blocks,
                                                NullRecorder&);
  void free(const Request& req, NullRecorder&);
  void reset_hashes(NullRecorder&);

  // Read-only: does not record a query.
  int peek_hit_tokens(const Request& req) const;

  bool can_allocate(const Request& req, int n_tokens,
                    const std::vector<KvBlock*>& hit_blocks) const;

  // Diagnostic counters (public for FieldSnapshot3 access).
  int queries = 0;
  int hits = 0;

  // Read-only views for state comparison in tests and debugging.
  const std::unordered_map<int, std::vector<KvBlock*>>& held() const { return held_; }
  const std::unordered_map<int, int>& n_cached() const { return n_cached_; }

 private:
  friend struct HeldBlocksRestore;
  friend struct HeldBlocksTruncate;
  friend struct CachedCountsRestore;

  template <typename Recorder>
  std::pair<std::vector<KvBlock*>, int> computed_blocks_impl(
      const Request& req, Recorder& rec);

  template <typename Recorder>
  std::optional<std::vector<KvBlock*>> allocate_impl(
      const Request& req, int n_tokens,
      const std::vector<KvBlock*>& hit_blocks, Recorder& rec);

  template <typename Recorder>
  void free_impl(const Request& req, Recorder& rec);

  template <typename Recorder>
  void reset_hashes_impl(Recorder& rec);

  int new_blocks_needed(const Request& req, int n_tokens,
                        const std::vector<KvBlock*>& hit_blocks) const;

  HeldBlocksRestore capture_held(int rid) const;
  HeldBlocksTruncate capture_held_length(int rid) const;

  int block_size_;
  bool enable_caching_;
  BlockPool pool_;
  std::unordered_map<int, std::vector<KvBlock*>> held_;
  std::unordered_map<int, int> n_cached_;
};

}  // namespace srl::engine
