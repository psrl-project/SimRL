// srl/component/rollout/engine_cpp/include/engine/pool.h
//
// The KV block pool: allocation, LRU reuse, and eviction.
//
// Port of srl/component/rollout/engine/kv/pool.py. The central invariant:
// a block may sit in the free list and in the hash map at the same time. That
// is the "nobody holds it, but it is still addressable" state. A hash is
// destroyed only when its block is handed to a new owner (_maybe_evict), never
// on release.
#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "engine/block.h"
#include "engine/free_list.h"
#include "engine/undo.h"

namespace srl::engine {

class BlockPool {
 public:
  // Args:
  //   num_blocks: Pool capacity.
  //   enable_caching: When false, blocks are still allocated and freed but no
  //       hash is ever recorded, so nothing can hit.
  BlockPool(int num_blocks, bool enable_caching = true);

  BlockPool(const BlockPool&) = delete;
  BlockPool& operator=(const BlockPool&) = delete;
  BlockPool(BlockPool&&) = delete;
  BlockPool& operator=(BlockPool&&) = delete;

  int num_blocks() const { return num_blocks_; }
  bool enable_caching() const { return enable_caching_; }
  int num_free_blocks() const { return free_.num_free(); }
  int num_used_blocks() const { return num_blocks_ - free_.num_free(); }

  // Fraction of the pool currently held, 0.0 for an empty pool.
  double usage() const {
    if (num_blocks_ <= 0) return 0.0;
    return static_cast<double>(num_used_blocks()) / num_blocks_;
  }

  // Free block ids, least recently freed first. O(num_free).
  std::vector<int> free_list_ids() const { return free_.ids(); }

  // The block holding `block_hash`, or nullptr on a miss.
  KvBlock* cached_block(const BlockHash& hash) const;

  // Read-only views for state comparison in tests and debugging.
  const std::vector<KvBlock>& blocks() const { return blocks_; }
  const std::unordered_map<BlockHash, KvBlock*, BlockHashHasher>& hash_map() const {
    return hash_to_block_;
  }

  // --- Recorded overloads (speculative / undo-tracked paths) ---

  std::vector<KvBlock*> new_blocks(int n, UndoLog& log);
  void touch(const std::vector<KvBlock*>& blocks, UndoLog& log);
  void free_blocks(const std::vector<KvBlock*>& blocks, UndoLog& log);
  void cache_full_blocks(const std::vector<KvBlock*>& blocks,
                         const std::vector<BlockHash>& hashes,
                         int n_cached, int n_full,
                         UndoLog& log, int hash_offset = 0);
  void reset_hashes(UndoLog& log);

  // --- NullRecorder overloads (committed paths) ---

  std::vector<KvBlock*> new_blocks(int n, NullRecorder&);
  void touch(const std::vector<KvBlock*>& blocks, NullRecorder&);
  void free_blocks(const std::vector<KvBlock*>& blocks, NullRecorder&);
  void cache_full_blocks(const std::vector<KvBlock*>& blocks,
                         const std::vector<BlockHash>& hashes,
                         int n_cached, int n_full,
                         NullRecorder&, int hash_offset = 0);
  void reset_hashes(NullRecorder&);

  int eviction_count = 0;

 private:
  friend struct HashMapBulkRestore;

  template <typename Recorder>
  std::vector<KvBlock*> new_blocks_impl(int n, Recorder& rec);

  template <typename Recorder>
  void touch_impl(const std::vector<KvBlock*>& blocks, Recorder& rec);

  template <typename Recorder>
  void free_blocks_impl(const std::vector<KvBlock*>& blocks, Recorder& rec);

  template <typename Recorder>
  void cache_full_blocks_impl(const std::vector<KvBlock*>& blocks,
                              const std::vector<BlockHash>& hashes,
                              int n_cached, int n_full,
                              Recorder& rec, int hash_offset);

  template <typename Recorder>
  void reset_hashes_impl(Recorder& rec);

  template <typename Recorder>
  bool maybe_evict(KvBlock* block, Recorder& rec);

  static std::vector<KvBlock> make_blocks(int n);

  int num_blocks_;
  bool enable_caching_;
  std::vector<KvBlock> blocks_;
  FreeBlockList free_;
  std::unordered_map<BlockHash, KvBlock*, BlockHashHasher> hash_to_block_;
};

}  // namespace srl::engine
