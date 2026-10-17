// srl/component/rollout/engine_cpp/bindings.cc
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "engine/cost_model.h"
#include "engine/event.h"
#include "engine/instance.h"
#include "engine/request.h"
#include "engine/version.h"

namespace py = pybind11;

namespace {

// CPython's `float // float`, step for step (`_float_div_mod` in
// `Objects/floatobject.c`). `std::floor(a / b)` disagrees with it near integers.
double py_floordiv(double vx, double wx) {
  double mod = std::fmod(vx, wx);
  double div = (vx - mod) / wx;
  if (mod != 0.0) {
    if ((wx < 0) != (mod < 0)) div -= 1.0;
  }
  double floordiv;
  if (div != 0.0) {
    floordiv = std::floor(div);
    if (div - floordiv > 0.5) floordiv += 1.0;
  } else {
    floordiv = std::copysign(0.0, vx / wx);
  }
  return floordiv;
}

// CPython's one-argument `round(float)`, which rounds half to even.
double py_round(double x) {
  double rounded = std::round(x);
  if (std::fabs(x - rounded) == 0.5) rounded = 2.0 * std::round(x / 2.0);
  return rounded;
}

// `int(x)` for the finite, non-negative values these keys take.
int64_t py_int(double x) { return static_cast<int64_t>(x); }

struct GemmKey {
  int64_t tokens;
  int64_t requests;
  bool operator==(const GemmKey& o) const {
    return tokens == o.tokens && requests == o.requests;
  }
};
struct GemmKeyHash {
  size_t operator()(const GemmKey& k) const {
    return std::hash<int64_t>{}(k.tokens * 1000003 + k.requests);
  }
};

// Mirrors `ComposedRollout._attention_key`, where context is
// `(seq_len, prefill_reqs)` and either part may be None.
struct AttentionKey {
  bool has_context;
  int64_t seq_len;
  int64_t prefill_reqs;
  bool has_generation;
  int64_t generation;
  int64_t requests;
  bool operator==(const AttentionKey& o) const {
    return has_context == o.has_context && seq_len == o.seq_len &&
           prefill_reqs == o.prefill_reqs && has_generation == o.has_generation &&
           generation == o.generation && requests == o.requests;
  }
};
struct AttentionKeyHash {
  size_t operator()(const AttentionKey& k) const {
    size_t h = std::hash<int64_t>{}(k.seq_len);
    auto mix = [&h](int64_t v) {
      h ^= std::hash<int64_t>{}(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    mix(k.prefill_reqs);
    mix(k.generation);
    mix(k.requests);
    mix((k.has_context ? 1 : 0) | (k.has_generation ? 2 : 0));
    return h;
  }
};

// Entries are exact, so dropping them only costs recomputation.
constexpr size_t kMaxTermEntries = size_t{1} << 18;

}  // namespace

// Trampoline: wraps any Python RolloutCostModel so C++ can call it.
//
// NOTE(claude): A model that publishes `cpp_term_spec()` has `step_time`
// assembled here from two term memos, so Python is called once per distinct key
// rather than once per step. Keys use CPython's float semantics and terms are
// summed in `predict`'s order, so the result is bit-identical to `predict`.
class PyRolloutCostModel : public srl::engine::RolloutCostModel {
 public:
  PyRolloutCostModel(py::object py_cost) : py_cost_(std::move(py_cost)) {
    py::object spec_fn = py::getattr(py_cost_, "cpp_term_spec", py::none());
    if (!spec_fn.is_none()) {
      py::object spec = spec_fn();
      if (!spec.is_none()) {
        term_fast_path_ = true;
        attention_layers_ = spec["attention_layers"].cast<int64_t>();
        gemm_fn_ = py_cost_.attr("_gemm_comm_sum");
        attention_fn_ = py_cost_.attr("_attention_from_key");
      }
    }
  }

  bool term_fast_path() const { return term_fast_path_; }

  // `phase` is unused because the Python model derives it from the step
  // arguments itself, and passing ours would let the two disagree.
  double predict(const std::string& /*phase*/, double n_tokens, double sum_l2,
                 double ctxsum, int n_reqs, double prefill_ctx) override {
    py::gil_scoped_acquire gil;
    return py_cost_
        .attr("step_time")(n_tokens, sum_l2, ctxsum, n_reqs, prefill_ctx)
        .cast<double>();
  }

  std::tuple<double, double, double> decode_coeffs(int n_d,
                                                   double ctxsum) override {
    py::gil_scoped_acquire gil;
    py::tuple result = py_cost_.attr("decode_coeffs")(n_d, ctxsum);
    return {result[0].cast<double>(), result[1].cast<double>(),
            result[2].cast<double>()};
  }

  bool saturated(double n_tokens, double t2, double ctxsum,
                 int n_reqs) override {
    py::gil_scoped_acquire gil;
    return py_cost_.attr("saturated")(n_tokens, t2, ctxsum, n_reqs)
        .cast<bool>();
  }

  // The segment queries are forwarded too. The C++ defaults rebuild them from
  // `decode_coeffs`, a secant approximation of what the Python model integrates.

  double step_time(double n_tokens, double t2, double ctxsum, int n_reqs,
                   double prefill_ctx) override {
    if (term_fast_path_) {
      double extent_sum = (prefill_ctx != 0.0) ? prefill_ctx : n_tokens;
      // Inputs Python would reject or divide by zero on go to Python, so it raises.
      bool invalid = n_tokens < 0 || t2 < 0 || ctxsum < 0 || n_reqs < 0 ||
                     prefill_ctx < 0 || (t2 > 0 && extent_sum == 0.0);
      if (!invalid) {
        return step_time_from_terms(n_tokens, t2, ctxsum, n_reqs, extent_sum);
      }
    }
    py::gil_scoped_acquire gil;
    return py_cost_
        .attr("step_time")(n_tokens, t2, ctxsum, n_reqs, prefill_ctx)
        .cast<double>();
  }

  double decode_segment_time(int n_d, double ctxsum, int k) override {
    py::gil_scoped_acquire gil;
    return py_cost_.attr("decode_segment_time")(n_d, ctxsum, k).cast<double>();
  }

  int decode_steps_within(int n_d, double ctxsum, double budget) override {
    py::gil_scoped_acquire gil;
    return py_cost_.attr("decode_steps_within")(n_d, ctxsum, budget).cast<int>();
  }

  int decode_saturated_steps(int n_d, double ctxsum, int k) override {
    py::gil_scoped_acquire gil;
    return py_cost_.attr("decode_saturated_steps")(n_d, ctxsum, k).cast<int>();
  }

 private:
  // `ComposedRollout.predict` for a dense model, whose expert term is zero.
  double step_time_from_terms(double n_tokens, double t2, double ctxsum,
                              int n_reqs, double extent_sum) {
    int64_t requests = std::max<int64_t>(n_reqs, 1);
    int64_t tokens = std::max<int64_t>(py_int(py_round(n_tokens)), 1);

    AttentionKey akey{false, 0, 0, false, 0, requests};
    if (t2 > 0) {
      akey.has_context = true;
      akey.seq_len = std::max<int64_t>(py_int(py_floordiv(t2, extent_sum)), 1);
      akey.prefill_reqs = std::max<int64_t>(
          py_int(py_round(std::pow(extent_sum, 2.0) / t2)), 1);
    }
    if (ctxsum > 0) {
      akey.has_generation = true;
      akey.generation = std::max<int64_t>(
          py_int(py_floordiv(ctxsum, static_cast<double>(requests))), 1);
    }

    double total = gemm_term(GemmKey{tokens, requests});
    total += static_cast<double>(attention_layers_) * attention_term(akey);
    return total;
  }

  double gemm_term(const GemmKey& key) {
    auto it = gemm_memo_.find(key);
    if (it != gemm_memo_.end()) return it->second;
    double value;
    {
      py::gil_scoped_acquire gil;
      value = gemm_fn_(key.tokens, key.requests).cast<double>();
    }
    if (gemm_memo_.size() >= kMaxTermEntries) gemm_memo_.clear();
    gemm_memo_.emplace(key, value);
    return value;
  }

  double attention_term(const AttentionKey& key) {
    auto it = attention_memo_.find(key);
    if (it != attention_memo_.end()) return it->second;
    double value;
    {
      py::gil_scoped_acquire gil;
      py::object context = key.has_context
                               ? py::object(py::make_tuple(key.seq_len, key.prefill_reqs))
                               : py::object(py::none());
      py::object generation = key.has_generation ? py::object(py::int_(key.generation))
                                                 : py::object(py::none());
      value = attention_fn_(context, generation, key.requests).cast<double>();
    }
    if (attention_memo_.size() >= kMaxTermEntries) attention_memo_.clear();
    attention_memo_.emplace(key, value);
    return value;
  }

  py::object py_cost_;
  bool term_fast_path_ = false;
  int64_t attention_layers_ = 0;
  py::object gemm_fn_;
  py::object attention_fn_;
  std::unordered_map<GemmKey, double, GemmKeyHash> gemm_memo_;
  std::unordered_map<AttentionKey, double, AttentionKeyHash> attention_memo_;
};

PYBIND11_MODULE(_engine_cpp, m) {
  m.doc() = "C++ rollout engine backend for SimRL";
  m.def("version", &srl::engine::version, "Engine version string");

  py::enum_<srl::engine::ReqStatus>(m, "ReqStatus")
      .value("QUEUED", srl::engine::ReqStatus::kQueued)
      .value("WAITING", srl::engine::ReqStatus::kWaiting)
      .value("RUNNING", srl::engine::ReqStatus::kRunning)
      .value("PREEMPTED", srl::engine::ReqStatus::kPreempted)
      .value("RECOMPUTING", srl::engine::ReqStatus::kRecomputing)
      .value("DONE", srl::engine::ReqStatus::kDone)
      .value("CONSUMED", srl::engine::ReqStatus::kConsumed)
      .value("DROPPED", srl::engine::ReqStatus::kDropped);

  py::class_<srl::engine::Segment>(m, "CppSegment")
      .def(py::init<int, int>(), py::arg("version"), py::arg("tokens") = 0)
      .def_readwrite("version", &srl::engine::Segment::version)
      .def_readwrite("tokens", &srl::engine::Segment::tokens);

  py::class_<srl::engine::Request,
             std::shared_ptr<srl::engine::Request>>(m, "CppRequest")
      .def(py::init([](int rid, int prompt_len, int target_len, int v_traj,
                       int group_id, int group_index, int source_group_id) {
             auto r = std::make_shared<srl::engine::Request>();
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
      .def_readwrite("rid", &srl::engine::Request::rid)
      .def_readwrite("prompt_len", &srl::engine::Request::prompt_len)
      .def_readwrite("target_len", &srl::engine::Request::target_len)
      .def_readwrite("v_traj", &srl::engine::Request::v_traj)
      .def_readwrite("group_id", &srl::engine::Request::group_id)
      .def_readwrite("group_index", &srl::engine::Request::group_index)
      .def_readwrite("source_group_id",
                     &srl::engine::Request::source_group_id)
      .def_readwrite("instance_id", &srl::engine::Request::instance_id)
      .def_readwrite("admit_time", &srl::engine::Request::admit_time)
      .def_readwrite("dispatch_time", &srl::engine::Request::dispatch_time)
      .def_readwrite("generated", &srl::engine::Request::generated)
      .def_readwrite("prefilled", &srl::engine::Request::prefilled)
      // Accept both the C++ ReqStatus and the Python ReqStatus by name.
      .def_property("status",
          [](const srl::engine::Request& self) { return self.status; },
          [](srl::engine::Request& self, py::object val) {
            try {
              self.status = val.cast<srl::engine::ReqStatus>();
              return;
            } catch (py::cast_error&) {}
            static const std::unordered_map<std::string,
                                            srl::engine::ReqStatus>
                name_map = {
                    {"QUEUED", srl::engine::ReqStatus::kQueued},
                    {"WAITING", srl::engine::ReqStatus::kWaiting},
                    {"RUNNING", srl::engine::ReqStatus::kRunning},
                    {"PREEMPTED", srl::engine::ReqStatus::kPreempted},
                    {"RECOMPUTING", srl::engine::ReqStatus::kRecomputing},
                    {"DONE", srl::engine::ReqStatus::kDone},
                    {"CONSUMED", srl::engine::ReqStatus::kConsumed},
                    {"DROPPED", srl::engine::ReqStatus::kDropped},
                };
            auto name = val.attr("name").cast<std::string>();
            auto it = name_map.find(name);
            if (it == name_map.end()) {
              throw py::value_error("Unknown ReqStatus: " + name);
            }
            self.status = it->second;
          })
      .def_readwrite("complete_time", &srl::engine::Request::complete_time)
      .def_readwrite("consume_version",
                     &srl::engine::Request::consume_version)
      .def_readwrite("num_aborts", &srl::engine::Request::num_aborts)
      .def_readwrite("num_preemptions",
                     &srl::engine::Request::num_preemptions)
      .def_readwrite("num_reroutes", &srl::engine::Request::num_reroutes)
      .def_readwrite("router_enqueue_time",
                     &srl::engine::Request::router_enqueue_time)
      .def_readwrite("reprefill_tokens",
                     &srl::engine::Request::reprefill_tokens)
      .def_readwrite("prefix_hit_tokens",
                     &srl::engine::Request::prefix_hit_tokens)
      .def_readwrite("segments", &srl::engine::Request::segments)
      .def_property_readonly("remaining", &srl::engine::Request::remaining)
      .def_property_readonly("ctx", &srl::engine::Request::ctx)
      .def_property_readonly("v_min", &srl::engine::Request::v_min)
      .def_property_readonly("v_max", &srl::engine::Request::v_max)
      .def("add_tokens", &srl::engine::Request::add_tokens)
      .def("staleness_true",
           [](const srl::engine::Request& self) -> int {
             if (!self.consume_version.has_value()) {
               throw std::runtime_error("Request was never consumed.");
             }
             return *self.consume_version - self.v_min();
           })
      .def("staleness_reported",
           [](const srl::engine::Request& self) -> int {
             if (!self.consume_version.has_value()) {
               throw std::runtime_error("Request was never consumed.");
             }
             return *self.consume_version - self.v_max();
           });

  py::enum_<srl::engine::EventKind>(m, "EventKind")
      .value("ITERATION_DONE", srl::engine::EventKind::kIterationDone)
      .value("REQUEST_COMPLETE", srl::engine::EventKind::kRequestComplete)
      .value("PREEMPT", srl::engine::EventKind::kPreempt);

  py::class_<srl::engine::EngineEvent>(m, "CppEngineEvent")
      .def(py::init<>())
      .def_readwrite("kind", &srl::engine::EngineEvent::kind)
      .def_readwrite("instance_id", &srl::engine::EngineEvent::instance_id)
      .def_readwrite("t", &srl::engine::EngineEvent::t)
      .def_readwrite("completed", &srl::engine::EngineEvent::completed)
      .def_readwrite("preempted", &srl::engine::EngineEvent::preempted);

  // --- StepStat ---

  py::class_<srl::engine::StepStat>(m, "CppStepStat")
      .def(py::init<>())
      .def_readwrite("n_tokens", &srl::engine::StepStat::n_tokens)
      .def_readwrite("n_decode", &srl::engine::StepStat::n_decode)
      .def_readwrite("n_prefill", &srl::engine::StepStat::n_prefill)
      .def_readwrite("saturated", &srl::engine::StepStat::saturated)
      .def_readwrite("steps", &srl::engine::StepStat::steps)
      .def_readwrite("saturated_steps",
                     &srl::engine::StepStat::saturated_steps)
      .def_readwrite("dt", &srl::engine::StepStat::dt)
      .def_readwrite("phase", &srl::engine::StepStat::phase)
      .def_readwrite("t", &srl::engine::StepStat::t)
      .def_readwrite("n_prefill_hit",
                     &srl::engine::StepStat::n_prefill_hit)
      .def_readwrite("n_prefill_computed",
                     &srl::engine::StepStat::n_prefill_computed);

  // --- Cost models ---

  py::class_<srl::engine::RolloutCostModel,
             std::shared_ptr<srl::engine::RolloutCostModel>>(
      m, "CppRolloutCostModel")
      .def("step_time", &srl::engine::RolloutCostModel::step_time,
           py::arg("n_tokens"), py::arg("t2"), py::arg("ctxsum"),
           py::arg("n_reqs") = 0, py::arg("prefill_ctx") = 0.0)
      .def("decode_segment_time",
           &srl::engine::RolloutCostModel::decode_segment_time)
      .def("decode_steps_within",
           &srl::engine::RolloutCostModel::decode_steps_within)
      .def("decode_saturated_steps",
           &srl::engine::RolloutCostModel::decode_saturated_steps);

  // Trampoline: wraps any Python RolloutCostModel so C++ NativeInstance can call it.
  py::class_<PyRolloutCostModel, srl::engine::RolloutCostModel,
             std::shared_ptr<PyRolloutCostModel>>(m, "PyWrappedCostModel")
      .def(py::init<py::object>(), py::arg("py_cost"))
      .def_property_readonly("term_fast_path",
                             &PyRolloutCostModel::term_fast_path);

  py::class_<srl::engine::PsrlCostModel,
             srl::engine::RolloutCostModel,
             std::shared_ptr<srl::engine::PsrlCostModel>>(
      m, "PsrlCostModel")
      .def(py::init<double, double, double, double, double, double, double,
                    double, double, double>(),
           py::arg("prefill_attn_b"), py::arg("prefill_attn_k"),
           py::arg("prefill_other_floor"), py::arg("prefill_other_b"),
           py::arg("prefill_other_k"), py::arg("decode_attn_b"),
           py::arg("decode_attn_k"), py::arg("decode_other_floor"),
           py::arg("decode_other_b"), py::arg("decode_other_k"));

  py::class_<srl::engine::UnifiedRooflineCostModel,
             srl::engine::RolloutCostModel,
             std::shared_ptr<srl::engine::UnifiedRooflineCostModel>>(
      m, "UnifiedRooflineCostModel")
      .def(py::init<double, double, double, double, double, int>(),
           py::arg("F"), py::arg("W") = 0.0, py::arg("G") = 0.0,
           py::arg("A_p") = 0.0, py::arg("A_d") = 0.0, py::arg("b") = 16);

  // --- Advance module functions ---

  m.def("crossing_step", &srl::engine::crossing_step, py::arg("F"),
        py::arg("alpha"), py::arg("beta"), py::arg("k_cap") = -1);
  m.def("elapsed", &srl::engine::elapsed, py::arg("F"), py::arg("alpha"),
        py::arg("beta"), py::arg("k"));
  m.def("steps_within", &srl::engine::steps_within, py::arg("F"),
        py::arg("alpha"), py::arg("beta"), py::arg("budget"));
  m.def("next_completion", &srl::engine::next_completion,
        py::arg("remaining"));

  // --- Instance ---

  py::class_<srl::engine::InstanceLoad>(m, "CppInstanceLoad")
      .def_readwrite("instance_id",
                     &srl::engine::InstanceLoad::instance_id)
      .def_readwrite("n_waiting", &srl::engine::InstanceLoad::n_waiting)
      .def_readwrite("n_running", &srl::engine::InstanceLoad::n_running)
      .def_readwrite("kv_utilization",
                     &srl::engine::InstanceLoad::kv_utilization)
      .def_readwrite("total_ctx", &srl::engine::InstanceLoad::total_ctx)
      .def_readwrite("tp", &srl::engine::InstanceLoad::tp)
      .def_property_readonly("n_active",
                             &srl::engine::InstanceLoad::n_active);

  py::class_<srl::engine::NativeInstance>(m, "CppInstance", py::dynamic_attr())
      .def(py::init([](std::shared_ptr<srl::engine::RolloutCostModel> cost,
                       int token_budget, int kv_blocks, int block_size,
                       int max_concurrency, bool enable_prefix_caching,
                       int instance_id, int tp, int max_running,
                       int chunk_cap, bool reject_if_kv_full,
                       bool reject_if_waiting,
                       bool reject_if_running_full) {
             return std::make_unique<srl::engine::NativeInstance>(
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
      // The whole KV state as plain Python data, so tests can compare it at block
      // level. Counters alone miss a wrong free-list order or a stale hash.
      .def("debug_kv_state",
           [](const srl::engine::NativeInstance& self) {
             const auto& kv = self.kv();
             const auto& pool = kv.pool();
             py::dict out;
             out["free_list"] = pool.free_list_ids();
             py::dict held;
             for (const auto& [rid, blocks] : kv.held()) {
               py::list ids;
               for (const auto* b : blocks) ids.append(b->block_id);
               held[py::int_(rid)] = ids;
             }
             out["held"] = held;
             py::dict n_cached;
             for (const auto& [rid, n] : kv.n_cached()) n_cached[py::int_(rid)] = n;
             out["n_cached"] = n_cached;
             py::dict hashes;
             for (const auto& [h, b] : pool.hash_map()) {
               hashes[py::make_tuple(h.is_shared, h.id, h.index)] = b->block_id;
             }
             out["hash_map"] = hashes;
             py::list blocks;
             for (const auto& b : pool.blocks()) {
               py::object hash = py::none();
               if (b.block_hash.has_value()) {
                 const auto& h = *b.block_hash;
                 hash = py::make_tuple(h.is_shared, h.id, h.index);
               }
               blocks.append(py::make_tuple(b.ref_cnt, hash));
             }
             out["blocks"] = blocks;
             out["eviction_count"] = pool.eviction_count;
             return out;
           })
      .def_property("t_local",
                    &srl::engine::NativeInstance::t_local,
                    &srl::engine::NativeInstance::set_t_local)
      // The strategy is the same Python object the native engine uses, so both
      // backends pick the same victim. It is also stored as `inst.eviction`,
      // which the router binds a valuator to. None restores the vLLM default.
      .def(
          "set_eviction",
          [](py::object self_obj, py::object strategy) {
            auto& self = self_obj.cast<srl::engine::NativeInstance&>();
            self_obj.attr("eviction") = strategy;
            if (strategy.is_none()) {
              self.set_eviction(nullptr);
              return;
            }
            self.set_eviction(
                [strategy](const std::vector<srl::engine::RequestPtr>& victims,
                           double now) -> srl::engine::RequestPtr {
                  py::gil_scoped_acquire gil;
                  py::object chosen = strategy.attr("select")(victims, now);
                  if (chosen.is_none()) return nullptr;
                  return chosen.cast<srl::engine::RequestPtr>();
                });
          },
          py::arg("strategy"))
      .def_property_readonly("instance_id",
                             &srl::engine::NativeInstance::instance_id)
      .def_property_readonly("tp", &srl::engine::NativeInstance::tp)
      .def_property_readonly("kv_blocks",
                             &srl::engine::NativeInstance::kv_blocks)
      .def_property_readonly(
          "block_size",
          [](const srl::engine::NativeInstance& self) {
            return self.block_size();
          })
      .def_property_readonly("n_preemptions",
                             &srl::engine::NativeInstance::n_preemptions)
      .def_property_readonly("kv_queries",
                             &srl::engine::NativeInstance::kv_queries)
      .def_property_readonly("kv_hits",
                             &srl::engine::NativeInstance::kv_hits)
      .def_property_readonly("has_pending_event",
                             &srl::engine::NativeInstance::has_pending_event)
      // NOTE(claude): pybind's `stl.h` converts a `std::vector` into a new list on
      // every read, whatever the return-value policy. Prefer the size queries and
      // `step_stats_since` below in any per-event path.
      .def_property_readonly("waiting",
                             &srl::engine::NativeInstance::waiting)
      .def_property_readonly("running",
                             &srl::engine::NativeInstance::running)
      .def_property_readonly("completed",
                             &srl::engine::NativeInstance::completed)
      .def_property_readonly("step_stats",
                             &srl::engine::NativeInstance::step_stats)
      // sizes without converting the vectors
      .def_property_readonly("n_active",
                             &srl::engine::NativeInstance::n_active)
      .def_property_readonly(
          "n_running",
          [](const srl::engine::NativeInstance& self) {
            return self.running().size();
          })
      .def_property_readonly(
          "n_waiting",
          [](const srl::engine::NativeInstance& self) {
            return self.waiting().size();
          })
      .def_property_readonly(
          "n_step_stats",
          [](const srl::engine::NativeInstance& self) {
            return self.step_stats().size();
          })
      // only the entries from `start` on, for a reader holding a cursor
      .def(
          "step_stats_since",
          [](const srl::engine::NativeInstance& self, std::size_t start) {
            const auto& all = self.step_stats();
            if (start >= all.size()) return std::vector<srl::engine::StepStat>{};
            return std::vector<srl::engine::StepStat>(all.begin() + start,
                                                         all.end());
          },
          py::arg("start"))
      .def("blocks_used", &srl::engine::NativeInstance::blocks_used)
      .def("free_blocks", &srl::engine::NativeInstance::free_blocks)
      .def("committed_blocks",
           &srl::engine::NativeInstance::committed_blocks)
      .def("get_load", &srl::engine::NativeInstance::get_load)
      .def("step_time_delta",
           &srl::engine::NativeInstance::step_time_delta)
      .def("peek_hit_tokens",
           &srl::engine::NativeInstance::peek_hit_tokens,
           py::arg("req"))
      .def("add_request", &srl::engine::NativeInstance::add_request,
           py::arg("req"), py::arg("t"))
      .def(
          "advance_one_event",
          [](srl::engine::NativeInstance& self,
             int version) -> py::object {
            auto ev = self.advance_one_event(version);
            if (!ev.has_value()) return py::none();
            return py::cast(*ev);
          },
          py::arg("version"))
      .def(
          "advance_iterations",
          [](srl::engine::NativeInstance& self, int version,
             bool first_speculative, bool iter_speculative, py::object top,
             double deadline) -> py::tuple {
            bool has_top = !top.is_none();
            double t_top = 0.0;
            int iid_top = 0;
            if (has_top) {
              auto pair = top.cast<py::tuple>();
              t_top = pair[0].cast<double>();
              iid_top = pair[1].cast<int>();
            }
            auto batch = self.advance_iterations(version, first_speculative,
                                                 iter_speculative, has_top,
                                                 t_top, iid_top, deadline);
            py::object pending = batch.pending.has_value()
                                     ? py::cast(*batch.pending)
                                     : py::object(py::none());
            return py::make_tuple(pending, batch.n_committed,
                                  batch.last_committed_t,
                                  batch.n_stats_committed);
          },
          py::arg("version"), py::arg("first_speculative"),
          py::arg("iter_speculative"), py::arg("top"), py::arg("deadline"))
      .def(
          "advance_one_event_committed",
          [](srl::engine::NativeInstance& self,
             int version) -> py::object {
            auto ev = self.advance_one_event_committed(version);
            if (!ev.has_value()) return py::none();
            return py::cast(*ev);
          },
          py::arg("version"))
      .def("commit_last_event",
           &srl::engine::NativeInstance::commit_last_event)
      .def("undo_last_event",
           &srl::engine::NativeInstance::undo_last_event)
      .def("abort_all", &srl::engine::NativeInstance::abort_all,
           py::arg("requeue") = true)
      .def("cancel", &srl::engine::NativeInstance::cancel,
           py::arg("reqs"), py::arg("keep_tokens") = true,
           py::arg("requeue") = true)
      .def(
          "advance_to",
          [](srl::engine::NativeInstance& self, double t_limit, int version,
             int max_steps) {
            return self.advance_to(t_limit, version, max_steps);
          },
          py::arg("t_limit"), py::arg("version"),
          py::arg("max_steps") = 100000);
}
