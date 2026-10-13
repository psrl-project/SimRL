// dryrun/component/rollout/engine_cpp/src/pool.cc
#include "engine/pool.h"

#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace dryrun::engine {

// --- Helpers ---

std::vector<KvBlock> BlockPool::make_blocks(int n) {
  std::vector<KvBlock> v;
  v.reserve(n);
  for (int i = 0; i < n; ++i) {
    v.emplace_back(i);
  }
  return v;
}

// --- Constructor ---

BlockPool::BlockPool(int num_blocks, bool enable_caching)
    : num_blocks_(num_blocks),
      enable_caching_(enable_caching),
      blocks_(make_blocks(num_blocks)),
      free_(blocks_) {
  if (num_blocks < 0) {
    throw std::invalid_argument(
        "Expected num_blocks >= 0, got " + std::to_string(num_blocks) + ".");
  }
}

// --- Read-only methods ---

KvBlock* BlockPool::cached_block(const BlockHash& hash) const {
  auto it = hash_to_block_.find(hash);
  return (it != hash_to_block_.end()) ? it->second : nullptr;
}

// --- Template implementations ---

template <typename Recorder>
std::vector<KvBlock*> BlockPool::new_blocks_impl(int n, Recorder& rec) {
  if (n > free_.num_free()) {
    throw std::invalid_argument(
        "Cannot allocate " + std::to_string(n) + " KV blocks: only " +
        std::to_string(free_.num_free()) + " free.");
  }
  std::vector<KvBlock*> taken;
  taken.reserve(n);
  for (int i = 0; i < n; ++i) {
    KvBlock* block = free_.popleft(rec);
    maybe_evict(block, rec);
    block->incr_ref();
    rec.record(RefIncr{block});
    taken.push_back(block);
  }
  return taken;
}

template <typename Recorder>
void BlockPool::touch_impl(const std::vector<KvBlock*>& blocks,
                           Recorder& rec) {
  for (KvBlock* block : blocks) {
    if (block->ref_cnt == 0) {
      free_.remove(block, rec);
    }
    block->incr_ref();
    rec.record(RefIncr{block});
  }
}

template <typename Recorder>
void BlockPool::free_blocks_impl(const std::vector<KvBlock*>& blocks,
                                 Recorder& rec) {
  for (KvBlock* block : blocks) {
    block->decr_ref();
    rec.record(RefDecr{block});
    if (block->ref_cnt == 0) {
      free_.append(block, rec);
    }
  }
}

template <typename Recorder>
void BlockPool::cache_full_blocks_impl(const std::vector<KvBlock*>& blocks,
                                       const std::vector<BlockHash>& hashes,
                                       int n_cached, int n_full,
                                       Recorder& rec, int hash_offset) {
  if (!enable_caching_ || n_cached >= n_full) return;

  int last = std::min({n_full,
                       hash_offset + static_cast<int>(hashes.size()),
                       static_cast<int>(blocks.size())});
  int start = std::max(n_cached, hash_offset);

  if constexpr (std::is_same_v<Recorder, UndoLog>) {
    // Build a HashMapChanged entry that notes each key BEFORE the write.
    HashMapChanged changed;
    changed.map = &hash_to_block_;
    for (int index = start; index < last; ++index) {
      KvBlock* block = blocks[index];
      if (!block->block_hash.has_value()) {
        const BlockHash& block_hash = hashes[index - hash_offset];
        auto it = hash_to_block_.find(block_hash);
        KvBlock* old =
            (it != hash_to_block_.end()) ? it->second : nullptr;
        changed.changes.push_back({block_hash, old});
        block->set_hash(block_hash);
        hash_to_block_[block_hash] = block;
        rec.record(HashAssigned{block});
      }
    }
    rec.record(std::move(changed));
  } else {
    // NullRecorder: do the work, skip undo bookkeeping.
    for (int index = start; index < last; ++index) {
      KvBlock* block = blocks[index];
      if (!block->block_hash.has_value()) {
        const BlockHash& block_hash = hashes[index - hash_offset];
        block->set_hash(block_hash);
        hash_to_block_[block_hash] = block;
      }
    }
  }
}

template <typename Recorder>
void BlockPool::reset_hashes_impl(Recorder& rec) {
  if (hash_to_block_.empty()) return;

  if constexpr (std::is_same_v<Recorder, UndoLog>) {
    rec.record(HashMapBulkRestore{this, hash_to_block_});
  }
  for (auto& [hash, block] : hash_to_block_) {
    block->reset_hash();
  }
  hash_to_block_.clear();
}

template <typename Recorder>
bool BlockPool::maybe_evict(KvBlock* block, Recorder& rec) {
  if (!block->block_hash.has_value()) return false;

  BlockHash evicted_hash = block->block_hash.value();

  // Restore the map pair only when this block is the one the map names.
  // See pool.py's _maybe_evict for the full rationale.
  auto it = hash_to_block_.find(evicted_hash);
  if (it != hash_to_block_.end() && it->second == block) {
    rec.record(HashMapEvicted{&hash_to_block_, evicted_hash, block});
    hash_to_block_.erase(it);
  } else {
    rec.record(HashMapCleared{block, evicted_hash});
  }

  // Capture eviction_count once per event for rollback.
  rec.record_once(
      JournalKey::of(this, JournalKey::kEvictionCount),
      [this]() -> UndoEntry {
        return FieldSnapshot3{this, FieldSnapshot3::Kind::kEvictionCount,
                              0.0, eviction_count, false, 0};
      });

  block->reset_hash();
  ++eviction_count;
  return true;
}

// --- Public overloads: UndoLog ---

std::vector<KvBlock*> BlockPool::new_blocks(int n, UndoLog& log) {
  return new_blocks_impl(n, log);
}

void BlockPool::touch(const std::vector<KvBlock*>& blocks, UndoLog& log) {
  touch_impl(blocks, log);
}

void BlockPool::free_blocks(const std::vector<KvBlock*>& blocks,
                            UndoLog& log) {
  free_blocks_impl(blocks, log);
}

void BlockPool::cache_full_blocks(const std::vector<KvBlock*>& blocks,
                                  const std::vector<BlockHash>& hashes,
                                  int n_cached, int n_full,
                                  UndoLog& log, int hash_offset) {
  cache_full_blocks_impl(blocks, hashes, n_cached, n_full, log, hash_offset);
}

void BlockPool::reset_hashes(UndoLog& log) {
  reset_hashes_impl(log);
}

// --- Public overloads: NullRecorder ---

std::vector<KvBlock*> BlockPool::new_blocks(int n, NullRecorder& nr) {
  return new_blocks_impl(n, nr);
}

void BlockPool::touch(const std::vector<KvBlock*>& blocks, NullRecorder& nr) {
  touch_impl(blocks, nr);
}

void BlockPool::free_blocks(const std::vector<KvBlock*>& blocks,
                            NullRecorder& nr) {
  free_blocks_impl(blocks, nr);
}

void BlockPool::cache_full_blocks(const std::vector<KvBlock*>& blocks,
                                  const std::vector<BlockHash>& hashes,
                                  int n_cached, int n_full,
                                  NullRecorder& nr, int hash_offset) {
  cache_full_blocks_impl(blocks, hashes, n_cached, n_full, nr, hash_offset);
}

void BlockPool::reset_hashes(NullRecorder& nr) {
  reset_hashes_impl(nr);
}

}  // namespace dryrun::engine
