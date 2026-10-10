// dryrun/component/rollout/engine_cpp/include/engine/free_list.h
#pragma once

#include <cassert>
#include <stdexcept>
#include <vector>

#include "engine/block.h"
#include "engine/undo.h"

namespace dryrun::engine {

class FreeBlockList {
 public:
  explicit FreeBlockList(std::vector<KvBlock>& blocks);

  int num_free() const { return num_free_; }

  std::vector<int> ids() const;

  KvBlock* popleft(UndoLog& log);
  void append(KvBlock* block, UndoLog& log);
  void remove(KvBlock* block, UndoLog& log);

  // NullRecorder overloads for committed paths.
  KvBlock* popleft(NullRecorder&);
  void append(KvBlock* block, NullRecorder&);
  void remove(KvBlock* block, NullRecorder&);

  // Undo helpers (called by journal entries).
  void relink(KvBlock* block, KvBlock* prev, KvBlock* next);
  void detach(KvBlock* block);

 private:
  void unlink(KvBlock* block);

  KvBlock head_{kSentinelId};
  KvBlock tail_{kSentinelId};
  int num_free_ = 0;
};

}  // namespace dryrun::engine
