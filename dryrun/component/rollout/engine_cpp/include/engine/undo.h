// dryrun/component/rollout/engine_cpp/include/engine/undo.h
#pragma once

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "engine/block.h"
#include "engine/request.h"

namespace dryrun::engine {

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
struct RequestSnapshot {
  RequestPtr req;
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

// Held-blocks restore for KvCacheManager.
struct HeldBlocksRestore {
  KvCacheManager* manager;
  int rid;
  bool existed;
  std::vector<KvBlock*> old_blocks;
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
    CachedCountsRestore
>;

// --- The undo log ---

class UndoLog {
 public:
  UndoLog() { entries_.reserve(256); }

  void record(UndoEntry e) { entries_.push_back(std::move(e)); }

  template <typename F>
  void record_once(uintptr_t key, F&& factory) {
    if (keys_.count(key)) return;
    keys_.insert(key);
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
  std::unordered_set<uintptr_t> keys_;
};

// Null recorder: does nothing. Used by committed paths.
struct NullRecorder {
  void record(UndoEntry) {}
  template <typename F>
  void record_once(uintptr_t, F&&) {}
};

inline NullRecorder kNullRecorder;

}  // namespace dryrun::engine
