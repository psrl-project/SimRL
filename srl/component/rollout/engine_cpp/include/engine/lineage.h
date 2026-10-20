// srl/component/rollout/engine_cpp/include/engine/lineage.h
//
// Block hashes derived from workload provenance.
//
// Port of srl/component/rollout/engine/kv/lineage.py. Two contexts agree on
// a prefix exactly when they agree on the provenance that produced it.
// Prompt blocks key on (is_shared=true, source_group_id, j). Response blocks
// key on (is_shared=false, rid, j).
#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/block.h"
#include "engine/request.h"

namespace srl::engine {

// How many full blocks this request's current context covers.
inline int block_count(const Request& req, int block_size) {
  return (req.prompt_len + req.generated) / block_size;
}

// Hash ids for full blocks [start, stop) of this request's context.
//
// A negative stop (the default) means "whole context". start is clamped to 0.
// Prompt blocks key on source_group_id. Response blocks key on rid.
inline std::vector<BlockHash> block_hashes(const Request& req, int block_size,
                                           int start = 0, int stop = -1) {
  if (block_size <= 0) {
    throw std::invalid_argument(
        "Expected block_size > 0, got " + std::to_string(block_size) + ".");
  }

  int n_blocks = (req.prompt_len + req.generated) / block_size;
  if (stop < 0 || stop > n_blocks) stop = n_blocks;
  if (start < 0) start = 0;
  if (start >= stop) return {};

  int shared_blocks = req.prompt_len / block_size;
  int group_id = req.source_group_id;
  int rid = req.rid;

  std::vector<BlockHash> result;
  result.reserve(stop - start);

  if (stop <= shared_blocks) {
    for (int j = start; j < stop; ++j) {
      result.push_back({true, group_id, j});
    }
  } else if (start >= shared_blocks) {
    for (int j = start; j < stop; ++j) {
      result.push_back({false, rid, j});
    }
  } else {
    for (int j = start; j < shared_blocks; ++j) {
      result.push_back({true, group_id, j});
    }
    for (int j = shared_blocks; j < stop; ++j) {
      result.push_back({false, rid, j});
    }
  }
  return result;
}

}  // namespace srl::engine
