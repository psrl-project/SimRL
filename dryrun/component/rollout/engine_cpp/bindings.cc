// dryrun/component/rollout/engine_cpp/bindings.cc
#include <pybind11/pybind11.h>
#include "engine/version.h"

namespace py = pybind11;

PYBIND11_MODULE(_engine_cpp, m) {
  m.doc() = "C++ rollout engine backend for DryRun-RL";
  m.def("version", &dryrun::engine::version, "Engine version string");
}
