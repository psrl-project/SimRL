// dryrun/component/rollout/engine_cpp/bindings.cc
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "engine/cost_model.h"
#include "engine/event.h"
#include "engine/instance.h"
#include "engine/request.h"
#include "engine/version.h"

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

  // --- Cost models ---

  py::class_<dryrun::engine::RolloutCostModel,
             std::shared_ptr<dryrun::engine::RolloutCostModel>>(
      m, "CppRolloutCostModel")
      .def("step_time", &dryrun::engine::RolloutCostModel::step_time,
           py::arg("n_tokens"), py::arg("t2"), py::arg("ctxsum"),
           py::arg("n_reqs") = 0, py::arg("prefill_ctx") = 0.0)
      .def("decode_segment_time",
           &dryrun::engine::RolloutCostModel::decode_segment_time)
      .def("decode_steps_within",
           &dryrun::engine::RolloutCostModel::decode_steps_within);

  py::class_<dryrun::engine::PsrlCostModel,
             dryrun::engine::RolloutCostModel,
             std::shared_ptr<dryrun::engine::PsrlCostModel>>(
      m, "PsrlCostModel")
      .def(py::init<double, double, double, double, double, double, double,
                    double, double, double>(),
           py::arg("prefill_attn_b"), py::arg("prefill_attn_k"),
           py::arg("prefill_other_floor"), py::arg("prefill_other_b"),
           py::arg("prefill_other_k"), py::arg("decode_attn_b"),
           py::arg("decode_attn_k"), py::arg("decode_other_floor"),
           py::arg("decode_other_b"), py::arg("decode_other_k"));

  py::class_<dryrun::engine::UnifiedRooflineCostModel,
             dryrun::engine::RolloutCostModel,
             std::shared_ptr<dryrun::engine::UnifiedRooflineCostModel>>(
      m, "UnifiedRooflineCostModel")
      .def(py::init<double, double, double, double, double, int>(),
           py::arg("F"), py::arg("W") = 0.0, py::arg("G") = 0.0,
           py::arg("A_p") = 0.0, py::arg("A_d") = 0.0, py::arg("b") = 16);

  // --- Advance module functions ---

  m.def("crossing_step", &dryrun::engine::crossing_step, py::arg("F"),
        py::arg("alpha"), py::arg("beta"), py::arg("k_cap") = -1);
  m.def("elapsed", &dryrun::engine::elapsed, py::arg("F"), py::arg("alpha"),
        py::arg("beta"), py::arg("k"));
  m.def("steps_within", &dryrun::engine::steps_within, py::arg("F"),
        py::arg("alpha"), py::arg("beta"), py::arg("budget"));
  m.def("next_completion", &dryrun::engine::next_completion,
        py::arg("remaining"));

  // --- Instance ---

  py::class_<dryrun::engine::InstanceLoad>(m, "CppInstanceLoad")
      .def_readwrite("instance_id",
                     &dryrun::engine::InstanceLoad::instance_id)
      .def_readwrite("n_waiting", &dryrun::engine::InstanceLoad::n_waiting)
      .def_readwrite("n_running", &dryrun::engine::InstanceLoad::n_running)
      .def_readwrite("kv_utilization",
                     &dryrun::engine::InstanceLoad::kv_utilization)
      .def_readwrite("total_ctx", &dryrun::engine::InstanceLoad::total_ctx)
      .def_readwrite("tp", &dryrun::engine::InstanceLoad::tp)
      .def_property_readonly("n_active",
                             &dryrun::engine::InstanceLoad::n_active);

  py::class_<dryrun::engine::NativeInstance>(m, "CppInstance")
      .def(py::init([](std::shared_ptr<dryrun::engine::RolloutCostModel> cost,
                       int token_budget, int kv_blocks, int block_size,
                       int max_concurrency, bool enable_prefix_caching,
                       int instance_id, int tp, int max_running,
                       int chunk_cap, bool reject_if_kv_full,
                       bool reject_if_waiting,
                       bool reject_if_running_full) {
             return std::make_unique<dryrun::engine::NativeInstance>(
                 std::move(cost), token_budget, kv_blocks, block_size,
                 max_running, max_concurrency, chunk_cap, instance_id, tp,
                 reject_if_kv_full, reject_if_waiting, reject_if_running_full,
                 enable_prefix_caching);
           }),
           py::arg("cost"), py::arg("token_budget"), py::arg("kv_blocks"),
           py::arg("block_size") = 16, py::arg("max_concurrency") = -1,
           py::arg("enable_prefix_caching") = true,
           py::arg("instance_id") = 0, py::arg("tp") = 1,
           py::arg("max_running") = -1, py::arg("chunk_cap") = -1,
           py::arg("reject_if_kv_full") = false,
           py::arg("reject_if_waiting") = false,
           py::arg("reject_if_running_full") = false)
      .def_property_readonly("t_local",
                             &dryrun::engine::NativeInstance::t_local)
      .def_property_readonly("instance_id",
                             &dryrun::engine::NativeInstance::instance_id)
      .def_property_readonly("tp", &dryrun::engine::NativeInstance::tp)
      .def_property_readonly("n_preemptions",
                             &dryrun::engine::NativeInstance::n_preemptions)
      .def_property_readonly("kv_queries",
                             &dryrun::engine::NativeInstance::kv_queries)
      .def_property_readonly("kv_hits",
                             &dryrun::engine::NativeInstance::kv_hits)
      .def_property_readonly("waiting",
                             &dryrun::engine::NativeInstance::waiting)
      .def_property_readonly("running",
                             &dryrun::engine::NativeInstance::running)
      .def_property_readonly("completed",
                             &dryrun::engine::NativeInstance::completed)
      .def("blocks_used", &dryrun::engine::NativeInstance::blocks_used)
      .def("free_blocks", &dryrun::engine::NativeInstance::free_blocks)
      .def("committed_blocks",
           &dryrun::engine::NativeInstance::committed_blocks)
      .def("get_load", &dryrun::engine::NativeInstance::get_load)
      .def("step_time_delta",
           &dryrun::engine::NativeInstance::step_time_delta)
      .def("add_request", &dryrun::engine::NativeInstance::add_request,
           py::arg("req"), py::arg("t"))
      .def(
          "advance_one_event",
          [](dryrun::engine::NativeInstance& self,
             int version) -> py::object {
            auto ev = self.advance_one_event(version);
            if (!ev.has_value()) return py::none();
            return py::cast(*ev);
          },
          py::arg("version"))
      .def("commit_last_event",
           &dryrun::engine::NativeInstance::commit_last_event)
      .def("undo_last_event",
           &dryrun::engine::NativeInstance::undo_last_event)
      .def("abort_all", &dryrun::engine::NativeInstance::abort_all,
           py::arg("requeue") = true)
      .def("cancel", &dryrun::engine::NativeInstance::cancel,
           py::arg("reqs"), py::arg("keep_tokens") = true,
           py::arg("requeue") = true)
      .def(
          "advance_to",
          [](dryrun::engine::NativeInstance& self, double t_limit, int version,
             int max_steps) {
            return self.advance_to(t_limit, version, max_steps);
          },
          py::arg("t_limit"), py::arg("version"),
          py::arg("max_steps") = 100000);
}
