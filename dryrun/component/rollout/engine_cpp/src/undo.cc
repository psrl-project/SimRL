// dryrun/component/rollout/engine_cpp/src/undo.cc
#include "engine/undo.h"
#include "engine/free_list.h"
#include "engine/instance.h"
#include "engine/kv_manager.h"
#include "engine/pool.h"

namespace dryrun::engine {

void UndoLog::rollback() {
  if (entries_.empty()) {
    throw std::runtime_error("Cannot roll back an empty undo log.");
  }
  for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
    std::visit([](auto& e) { e.restore(); }, *it);
  }
  clear();
}

void FreeListRelink::restore() {
  list->relink(block, prev, next);
}

void FreeListDetach::restore() {
  list->detach(block);
}

void HashMapChanged::restore() {
  for (auto it = changes.rbegin(); it != changes.rend(); ++it) {
    if (it->old_block == nullptr) {
      map->erase(it->key);
    } else {
      (*map)[it->key] = it->old_block;
    }
  }
}

RequestSnapshot RequestSnapshot::capture(const RequestPtr& r) {
  return {
      r,
      r->prefilled,
      r->generated,
      r->status,
      r->complete_time,
      r->num_preemptions,
      r->num_aborts,
      r->reprefill_tokens,
      r->prefix_hit_tokens,
      r->segments.empty() ? 0 : r->segments.back().tokens,
      r->segments.size(),
  };
}

void RequestSnapshot::restore() {
  req->prefilled = prefilled;
  req->generated = generated;
  req->status = status;
  req->complete_time = complete_time;
  req->num_preemptions = num_preemptions;
  req->num_aborts = num_aborts;
  req->reprefill_tokens = reprefill_tokens;
  req->prefix_hit_tokens = prefix_hit_tokens;
  req->segments.resize(seg_count);
  if (!req->segments.empty()) {
    req->segments.back().tokens = seg_tail_tokens;
  }
}

// --- FieldSnapshot3: kEvictionCount and kCounters are Task 4, kInstanceScalars is Task 5 ---

void FieldSnapshot3::restore() {
  switch (kind) {
    case Kind::kEvictionCount: {
      auto* pool = static_cast<BlockPool*>(target);
      pool->eviction_count = i0;
      break;
    }
    case Kind::kCounters: {
      auto* mgr = static_cast<KvCacheManager*>(target);
      mgr->queries = i0;
      mgr->hits = i1;
      break;
    }
    case Kind::kInstanceScalars: {
      auto* inst = static_cast<NativeInstance*>(target);
      // Fields stored: d0 = t_local_, i0 = n_preemptions_,
      // b0 = last_decode_blocked_.
      inst->t_local_ = d0;
      inst->n_preemptions_ = i0;
      inst->last_decode_blocked_ = b0;
      break;
    }
  }
}

void VectorRestore::restore() {
  auto* inst = static_cast<NativeInstance*>(instance);
  if (which == Which::kWaiting) {
    inst->waiting_ = std::move(old_list);
  } else {
    inst->running_ = std::move(old_list);
  }
}

void VectorTruncate::restore() {
  auto* inst = static_cast<NativeInstance*>(target);
  if (which == Which::kCompleted) {
    inst->completed_.resize(old_size);
  } else if (which == Which::kStepStats) {
    inst->step_stats_.resize(old_size);
  } else {
    // kSegments is handled by RequestSnapshot::restore.
    throw std::runtime_error(
        "VectorTruncate::restore(kSegments) should be handled by "
        "RequestSnapshot.");
  }
}

void HashMapBulkRestore::restore() {
  // Merge old entries back into the pool's map (Python: dict.update).
  for (auto& [hash, block] : old_map) {
    pool->hash_to_block_[hash] = block;
  }
  // Restore each block's own hash field.
  for (auto& [hash, block] : old_map) {
    block->reset_hash();
    block->set_hash(hash);
  }
}

void HeldBlocksRestore::restore() {
  if (existed) {
    manager->held_[rid] = std::move(old_blocks);
  } else {
    manager->held_.erase(rid);
  }
  if (old_n_cached.has_value()) {
    manager->n_cached_[rid] = old_n_cached.value();
  } else {
    manager->n_cached_.erase(rid);
  }
}

void CachedCountsRestore::restore() {
  for (auto& [k, v] : old_counts) {
    manager->n_cached_[k] = v;
  }
}

}  // namespace dryrun::engine
