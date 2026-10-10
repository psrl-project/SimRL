// dryrun/component/rollout/engine_cpp/include/engine/event.h
#pragma once

#include <vector>
#include "engine/request.h"

namespace dryrun::engine {

enum class EventKind {
  kIterationDone,
  kRequestComplete,
  kPreempt,
};

struct EngineEvent {
  EventKind kind = EventKind::kIterationDone;
  int instance_id = 0;
  double t = 0.0;
  std::vector<RequestPtr> completed;
  std::vector<RequestPtr> preempted;
};

}  // namespace dryrun::engine
