// srl/component/rollout/engine_cpp/include/engine/undo.h
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "engine/block.h"
#include "engine/request.h"

namespace srl::engine {

// Forward declarations for types that undo entries reference.
class FreeBlockList;
class BlockPool;
class KvCacheManager;

// --- Journal entry types ---

// Restores scalar fields to their pre-event values.
// Stores up to 3 field values inline (covers all call sites).
struct FieldSnapshot3 {
  void* target;
  // Offsets into the target struct, and old values.
  // Using a union approach: store as uint64_t and reinterpret on restore.
  // Simpler approach: store typed old values for the known field combos.
  // Since every call site is known, we use an enum to tag which fields.
  enum class Kind : uint8_t {
    kInstanceScalars,     // t_local_, n_preemptions_, last_decode_blocked_
    kCounters,            // queries, hits (KvCacheManager)
    kEvictionCount,       // eviction_count (BlockPool)
  };
  Kind kind;
  double d0;
  int i0;
  bool b0;
  int i1;

  void restore();
};

// Restores waiting/running lists to their pre-event state (shallow copy).
struct VectorRestore {
  enum class Which : uint8_t { kWaiting, kRunning };
  void* instance;
  Which which;
  std::vector<RequestPtr> old_list;

  void restore();
};

// Truncates an append-only vector back to its pre-event length.
struct VectorTruncate {
  enum class Which : uint8_t { kCompleted, kStepStats, kSegments };
  void* target;
  Which which;
  size_t old_size;

  void restore();
};

// Restores Request mutable fields to their pre-event values.
//
// A raw pointer is safe because the same log holds a `VectorRestore` copy of
// both resident lists, which keeps every snapshotted request alive.
struct RequestSnapshot {
  Request* req;
  int prefilled;
  int generated;
  ReqStatus status;
  std::optional<double> complete_time;
  int num_preemptions;
  int num_aborts;
  int reprefill_tokens;
  int prefix_hit_tokens;
  // Last segment tail token count (if segments existed).
  int seg_tail_tokens;
  size_t seg_count;

  static RequestSnapshot capture(const RequestPtr& r);
  void restore();
};

// Free-list journal: re-link an unlinked block.
struct FreeListRelink {
  FreeBlockList* list;
  KvBlock* block;
  KvBlock* prev;
  KvBlock* next;

  void restore();
};

// Free-list journal: detach an appended block.
struct FreeListDetach {
  FreeBlockList* list;
  KvBlock* block;

  void restore();
};

// Refcount journals.
struct RefIncr {
  KvBlock* block;
  void restore() { block->decr_ref(); }
};

struct RefDecr {
  KvBlock* block;
  void restore() { block->incr_ref(); }
};

// Hash journals.
struct HashAssigned {
  KvBlock* block;
  void restore() { block->reset_hash(); }
};

struct HashMapEvicted {
  std::unordered_map<BlockHash, KvBlock*, BlockHashHasher>* map;
  BlockHash key;
  KvBlock* block;

  void restore() {
    (*map)[key] = block;
    block->reset_hash();
    block->set_hash(key);
  }
};

struct HashMapCleared {
  KvBlock* block;
  BlockHash key;

  void restore() {
    block->reset_hash();
    block->set_hash(key);
  }
};

// MappingKeysChanged equivalent: restore displaced values.
struct HashMapChanged {
  struct Entry {
    BlockHash key;
    KvBlock* old_block;  // nullptr = was absent (MISSING)
  };
  std::unordered_map<BlockHash, KvBlock*, BlockHashHasher>* map;
  std::vector<Entry> changes;

  void restore();
};

// Bulk restore after reset_hashes.
struct HashMapBulkRestore {
  BlockPool* pool;
  std::unordered_map<BlockHash, KvBlock*, BlockHashHasher> old_map;

  void restore();
};

// Held-blocks restore for KvCacheManager: the whole list, for a free.
struct HeldBlocksRestore {
  KvCacheManager* manager;
  int rid;
  bool existed;
  std::vector<KvBlock*> old_blocks;
  std::optional<int> old_n_cached;

  void restore();
};

// Held-blocks restore for an allocation. An allocation only appends to the held
// list, so cutting it back to its old length undoes it without copying the list.
struct HeldBlocksTruncate {
  KvCacheManager* manager;
  int rid;
  bool existed;
  size_t old_size;
  std::optional<int> old_n_cached;

  void restore();
};

// Cached-counts restore after reset_hashes.
struct CachedCountsRestore {
  KvCacheManager* manager;
  std::unordered_map<int, int> old_counts;

  void restore();
};

// --- The variant type ---

using UndoEntry = std::variant<
    FieldSnapshot3,
    VectorRestore,
    VectorTruncate,
    RequestSnapshot,
    FreeListRelink,
    FreeListDetach,
    RefIncr,
    RefDecr,
    HashAssigned,
    HashMapEvicted,
    HashMapCleared,
    HashMapChanged,
    HashMapBulkRestore,
    HeldBlocksRestore,
    HeldBlocksTruncate,
    CachedCountsRestore
>;

// Identity of one piece of state a `record_once` entry protects: the object
// that owns it, and which part of that object.
//
// NOTE(claude): Keep owner and part separate. An object's address is also the
// address of its first member, so a key folded into one integer lets two
// different pieces of state collide and the second is silently not journaled.
struct JournalKey {
  enum Part : uint32_t {
    kRequestFields = 1,
    kKvCounters = 2,
    kHeldBlocks = 3,
    kEvictionCount = 4,
  };
  const void* owner;
  uint64_t part;

  static JournalKey of(const void* owner, Part kind, uint32_t id = 0) {
    return {owner, (static_cast<uint64_t>(kind) << 32) | id};
  }
  bool operator==(const JournalKey& o) const {
    return owner == o.owner && part == o.part;
  }
};

// Open-addressing set of journal keys. `record_once` consults it for every
// resident on every event, so clearing bumps a generation stamp in O(1).
class FlatKeySet {
 public:
  FlatKeySet() { rehash(1024); }

  // Returns true when `key` was absent and has now been added.
  bool insert(const JournalKey& key) {
    if ((size_ + 1) * 2 > keys_.size()) rehash(keys_.size() * 2);
    const size_t mask = keys_.size() - 1;
    size_t i = mix(key) & mask;
    while (stamp_[i] == gen_) {
      if (keys_[i] == key) return false;
      i = (i + 1) & mask;
    }
    stamp_[i] = gen_;
    keys_[i] = key;
    ++size_;
    return true;
  }

  void clear() {
    size_ = 0;
    if (++gen_ == 0) {
      std::fill(stamp_.begin(), stamp_.end(), 0u);
      gen_ = 1;
    }
  }

 private:
  static size_t mix(const JournalKey& k) {
    uint64_t x = reinterpret_cast<uintptr_t>(k.owner) ^ (k.part * 0x9e3779b97f4a7c15ULL);
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return static_cast<size_t>(x);
  }

  void rehash(size_t capacity) {
    std::vector<JournalKey> old_keys = std::move(keys_);
    std::vector<uint32_t> old_stamp = std::move(stamp_);
    const uint32_t old_gen = gen_;
    keys_.assign(capacity, JournalKey{nullptr, 0});
    stamp_.assign(capacity, 0u);
    gen_ = 1;
    size_ = 0;
    for (size_t j = 0; j < old_keys.size(); ++j) {
      if (old_stamp[j] == old_gen) insert(old_keys[j]);
    }
  }

  std::vector<JournalKey> keys_;
  std::vector<uint32_t> stamp_;
  uint32_t gen_ = 1;
  size_t size_ = 0;
};

// --- The undo log ---

class UndoLog {
 public:
  UndoLog() { entries_.reserve(256); }

  void record(UndoEntry e) { entries_.push_back(std::move(e)); }

  // Build and record an entry. The factory lets the null recorder skip
  // constructing one.
  template <typename F>
  void record_lazy(F&& factory) {
    entries_.push_back(factory());
  }

  template <typename F>
  void record_once(const JournalKey& key, F&& factory) {
    if (!keys_.insert(key)) return;
    entries_.push_back(factory());
  }

  void rollback();
  void clear() {
    entries_.clear();
    keys_.clear();
  }

  size_t size() const { return entries_.size(); }
  bool empty() const { return entries_.empty(); }

 private:
  std::vector<UndoEntry> entries_;
  FlatKeySet keys_;
};

// Null recorder: does nothing. Used by committed paths.
struct NullRecorder {
  void record(UndoEntry) {}
  template <typename F>
  void record_lazy(F&&) {}
  template <typename F>
  void record_once(const JournalKey&, F&&) {}
};

inline NullRecorder kNullRecorder;

}  // namespace srl::engine
