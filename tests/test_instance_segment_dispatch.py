"""`instance.py` must take its decode-segment quantities from the cost model."""

from __future__ import annotations

from dryrun.component.rollout.engine import NativeInstance, Request
from dryrun.cost.rollout import PSRLFitted


def _psrl() -> PSRLFitted:
    return PSRLFitted(
        prefill_attn_b=2.0e-4,
        prefill_attn_k=3.0e-9,
        prefill_other_floor=1.0e-3,
        prefill_other_b=5.0e-4,
        prefill_other_k=2.0e-7,
        decode_attn_b=1.0e-4,
        decode_attn_k=4.0e-9,
        decode_other_floor=2.0e-3,
        decode_other_b=6.0e-4,
        decode_other_k=1.0e-5,
    )


class _RecordingCost(PSRLFitted):
    """A PSRL model that records which segment methods the simulator called."""

    def __init__(self) -> None:
        base = _psrl()
        super().__init__(**{name: getattr(base, name) for name in (
            "prefill_attn_b", "prefill_attn_k", "prefill_other_floor",
            "prefill_other_b", "prefill_other_k", "decode_attn_b", "decode_attn_k",
            "decode_other_floor", "decode_other_b", "decode_other_k",
        )})
        self.calls: list[str] = []

    def decode_segment_time(self, n_d, ctxsum, k):
        self.calls.append("decode_segment_time")
        return super().decode_segment_time(n_d, ctxsum, k)

    def decode_steps_within(self, n_d, ctxsum, budget):
        self.calls.append("decode_steps_within")
        return super().decode_steps_within(n_d, ctxsum, budget)

    def decode_saturated_steps(self, n_d, ctxsum, k):
        self.calls.append("decode_saturated_steps")
        return super().decode_saturated_steps(n_d, ctxsum, k)


def _instance(cost) -> NativeInstance:
    inst = NativeInstance(cost, token_budget=8192, kv_blocks=4000, block_size=16, instance_id=0)
    for rid in range(4):
        inst.add_request(Request(rid=rid, prompt_len=256, target_len=60, v_traj=0, admit_time=0.0), 0.0)
    return inst


def test_a_decode_event_asks_the_model_for_its_segment():
    cost = _RecordingCost()
    inst = _instance(cost)
    for _ in range(30):
        if inst.advance_one_event(0) is None:
            break
    assert "decode_segment_time" in cost.calls
    assert "decode_saturated_steps" in cost.calls


def test_a_time_limited_decode_asks_the_model_how_many_steps_fit():
    cost = _RecordingCost()
    inst = _instance(cost)
    inst.run_until(0.05, 0)
    assert "decode_steps_within" in cost.calls
    assert "decode_segment_time" in cost.calls


def test_an_overriding_model_changes_the_reported_segment_time():
    """A model that halves its segment time must halve the simulator's elapsed time."""

    class _Halved(_RecordingCost):
        def decode_segment_time(self, n_d, ctxsum, k):
            return 0.5 * super().decode_segment_time(n_d, ctxsum, k)

    plain, halved = _instance(_RecordingCost()), _instance(_Halved())
    for inst in (plain, halved):
        for _ in range(30):
            if inst.advance_one_event(0) is None:
                break
    plain_decode = sum(s.dt for s in plain.step_stats if s.phase == "decode")
    halved_decode = sum(s.dt for s in halved.step_stats if s.phase == "decode")
    assert plain_decode > 0
    assert halved_decode == 0.5 * plain_decode
