// dryrun/component/rollout/engine_cpp/bindings.cc
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "engine/version.h"
#include "engine/request.h"
#include "engine/event.h"

namespace py = pybind11;

PYBIND11_MODULE(_engine_cpp, m) {
  m.doc() = "C++ rollout engine backend for DryRun-RL";
  m.def("version", &dryrun::engine::version, "Engine version string");

  py::enum_<dryrun::engine::ReqStatus>(m, "ReqStatus")
      .value("QUEUED", dryrun::engine::ReqStatus::kQueued)
      .value("WAITING", dryrun::engine::ReqStatus::kWaiting)
      .value("RUNNING", dryrun::engine::ReqStatus::kRunning)
      .value("PREEMPTED", dryrun::engine::ReqStatus::kPreempted)
      .value("RECOMPUTING", dryrun::engine::ReqStatus::kRecomputing)
      .value("DONE", dryrun::engine::ReqStatus::kDone)
      .value("CONSUMED", dryrun::engine::ReqStatus::kConsumed)
      .value("DROPPED", dryrun::engine::ReqStatus::kDropped);

  py::class_<dryrun::engine::Segment>(m, "CppSegment")
      .def(py::init<int, int>(), py::arg("version"), py::arg("tokens") = 0)
      .def_readwrite("version", &dryrun::engine::Segment::version)
      .def_readwrite("tokens", &dryrun::engine::Segment::tokens);

  py::class_<dryrun::engine::Request,
             std::shared_ptr<dryrun::engine::Request>>(m, "CppRequest")
      .def(py::init([](int rid, int prompt_len, int target_len, int v_traj,
                       int group_id, int group_index, int source_group_id) {
             auto r = std::make_shared<dryrun::engine::Request>();
             r->rid = rid;
             r->prompt_len = prompt_len;
             r->target_len = target_len;
             r->v_traj = v_traj;
             r->group_id = group_id;
             r->group_index = group_index;
             r->source_group_id = source_group_id;
             return r;
           }),
           py::arg("rid"), py::arg("prompt_len"), py::arg("target_len"),
           py::arg("v_traj"), py::arg("group_id") = 0,
           py::arg("group_index") = 0, py::arg("source_group_id") = 0)
      .def_readwrite("rid", &dryrun::engine::Request::rid)
      .def_readwrite("prompt_len", &dryrun::engine::Request::prompt_len)
      .def_readwrite("target_len", &dryrun::engine::Request::target_len)
      .def_readwrite("v_traj", &dryrun::engine::Request::v_traj)
      .def_readwrite("group_id", &dryrun::engine::Request::group_id)
      .def_readwrite("group_index", &dryrun::engine::Request::group_index)
      .def_readwrite("source_group_id",
                     &dryrun::engine::Request::source_group_id)
      .def_readwrite("instance_id", &dryrun::engine::Request::instance_id)
      .def_readwrite("admit_time", &dryrun::engine::Request::admit_time)
      .def_readwrite("dispatch_time", &dryrun::engine::Request::dispatch_time)
      .def_readwrite("generated", &dryrun::engine::Request::generated)
      .def_readwrite("prefilled", &dryrun::engine::Request::prefilled)
      .def_readwrite("status", &dryrun::engine::Request::status)
      .def_readwrite("complete_time", &dryrun::engine::Request::complete_time)
      .def_readwrite("consume_version",
                     &dryrun::engine::Request::consume_version)
      .def_readwrite("num_aborts", &dryrun::engine::Request::num_aborts)
      .def_readwrite("num_preemptions",
                     &dryrun::engine::Request::num_preemptions)
      .def_readwrite("num_reroutes", &dryrun::engine::Request::num_reroutes)
      .def_readwrite("router_enqueue_time",
                     &dryrun::engine::Request::router_enqueue_time)
      .def_readwrite("reprefill_tokens",
                     &dryrun::engine::Request::reprefill_tokens)
      .def_readwrite("prefix_hit_tokens",
                     &dryrun::engine::Request::prefix_hit_tokens)
      .def_readwrite("segments", &dryrun::engine::Request::segments)
      .def_property_readonly("remaining", &dryrun::engine::Request::remaining)
      .def_property_readonly("ctx", &dryrun::engine::Request::ctx)
      .def_property_readonly("v_min", &dryrun::engine::Request::v_min)
      .def_property_readonly("v_max", &dryrun::engine::Request::v_max)
      .def("add_tokens", &dryrun::engine::Request::add_tokens);

  py::enum_<dryrun::engine::EventKind>(m, "EventKind")
      .value("ITERATION_DONE", dryrun::engine::EventKind::kIterationDone)
      .value("REQUEST_COMPLETE", dryrun::engine::EventKind::kRequestComplete)
      .value("PREEMPT", dryrun::engine::EventKind::kPreempt);

  py::class_<dryrun::engine::EngineEvent>(m, "CppEngineEvent")
      .def(py::init<>())
      .def_readwrite("kind", &dryrun::engine::EngineEvent::kind)
      .def_readwrite("instance_id", &dryrun::engine::EngineEvent::instance_id)
      .def_readwrite("t", &dryrun::engine::EngineEvent::t)
      .def_readwrite("completed", &dryrun::engine::EngineEvent::completed)
      .def_readwrite("preempted", &dryrun::engine::EngineEvent::preempted);
}
