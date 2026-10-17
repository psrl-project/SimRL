// srl/component/rollout/engine_cpp/src/free_list.cc
#include "engine/free_list.h"

namespace srl::engine {

FreeBlockList::FreeBlockList(std::vector<KvBlock>& blocks)
    : num_free_(static_cast<int>(blocks.size())) {
  KvBlock* prev = &head_;
  for (auto& block : blocks) {
    prev->next_free = &block;
    block.prev_free = prev;
    prev = &block;
  }
  prev->next_free = &tail_;
  tail_.prev_free = prev;
}

std::vector<int> FreeBlockList::ids() const {
  std::vector<int> out;
  auto* node = head_.next_free;
  while (node != nullptr && node != &tail_) {
    out.push_back(node->block_id);
    node = node->next_free;
  }
  return out;
}

void FreeBlockList::unlink(KvBlock* block) {
  auto* prev = block->prev_free;
  auto* next = block->next_free;
  assert(prev && next);
  prev->next_free = next;
  next->prev_free = prev;
  block->prev_free = nullptr;
  block->next_free = nullptr;
  --num_free_;
}

KvBlock* FreeBlockList::popleft(UndoLog& log) {
  auto* block = head_.next_free;
  if (!block || block == &tail_) {
    throw std::runtime_error("Cannot pop from an empty free list.");
  }
  auto* prev = block->prev_free;
  auto* next = block->next_free;
  unlink(block);
  log.record(FreeListRelink{this, block, prev, next});
  return block;
}

void FreeBlockList::append(KvBlock* block, UndoLog& log) {
  auto* last = tail_.prev_free;
  assert(last);
  last->next_free = block;
  block->prev_free = last;
  block->next_free = &tail_;
  tail_.prev_free = block;
  ++num_free_;
  log.record(FreeListDetach{this, block});
}

void FreeBlockList::remove(KvBlock* block, UndoLog& log) {
  if (!block->prev_free || !block->next_free) {
    throw std::runtime_error("Block is not in the free list.");
  }
  auto* prev = block->prev_free;
  auto* next = block->next_free;
  unlink(block);
  log.record(FreeListRelink{this, block, prev, next});
}

// NullRecorder overloads — identical mutations, no journaling.
KvBlock* FreeBlockList::popleft(NullRecorder&) {
  auto* block = head_.next_free;
  if (!block || block == &tail_) {
    throw std::runtime_error("Cannot pop from an empty free list.");
  }
  unlink(block);
  return block;
}

void FreeBlockList::append(KvBlock* block, NullRecorder&) {
  auto* last = tail_.prev_free;
  assert(last);
  last->next_free = block;
  block->prev_free = last;
  block->next_free = &tail_;
  tail_.prev_free = block;
  ++num_free_;
}

void FreeBlockList::remove(KvBlock* block, NullRecorder&) {
  if (!block->prev_free || !block->next_free) {
    throw std::runtime_error("Block is not in the free list.");
  }
  unlink(block);
}

void FreeBlockList::relink(KvBlock* block, KvBlock* prev, KvBlock* next) {
  prev->next_free = block;
  next->prev_free = block;
  block->prev_free = prev;
  block->next_free = next;
  ++num_free_;
}

void FreeBlockList::detach(KvBlock* block) {
  auto* prev = block->prev_free;
  auto* next = block->next_free;
  assert(prev && next);
  prev->next_free = next;
  next->prev_free = prev;
  block->prev_free = nullptr;
  block->next_free = nullptr;
  --num_free_;
}

}  // namespace srl::engine
