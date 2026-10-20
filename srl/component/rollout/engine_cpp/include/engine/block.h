// srl/component/rollout/engine_cpp/include/engine/block.h
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>

namespace srl::engine {

struct BlockHash {
  bool is_shared;  // true = prompt block ("p"), false = private ("r")
  int id;          // source_group_id or rid
  int index;       // block index j

  bool operator==(const BlockHash& o) const {
    return is_shared == o.is_shared && id == o.id && index == o.index;
  }
};

struct BlockHashHasher {
  size_t operator()(const BlockHash& h) const {
    // FNV-1a inspired mix, simple and effective for 3 small integers.
    size_t seed = static_cast<size_t>(h.is_shared);
    seed ^= std::hash<int>{}(h.id) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>{}(h.index) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};

static constexpr int kSentinelId = -1;

struct KvBlock {
  int block_id;
  int ref_cnt = 0;
  std::optional<BlockHash> block_hash;
  KvBlock* prev_free = nullptr;
  KvBlock* next_free = nullptr;

  explicit KvBlock(int id) : block_id(id) {}

  void incr_ref() { ++ref_cnt; }
  void decr_ref() { --ref_cnt; }

  void set_hash(const BlockHash& h) {
    if (block_hash.has_value()) {
      throw std::runtime_error(
          "Block " + std::to_string(block_id) + " already holds a hash; "
          "call reset_hash() before assigning.");
    }
    block_hash = h;
  }

  void reset_hash() { block_hash.reset(); }
};

}  // namespace srl::engine
