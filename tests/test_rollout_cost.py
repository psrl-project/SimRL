"""Synthetic regression tests for all rollout cost model families."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from srl.cost.artifact import CostArtifact
from srl.cost.rollout import (
    DistServe,
    PSRLFitted,
    RolloutWorkload,
    UnifiedRoofline,
)
from srl.cost.rollout.plot import plot_vllm_rollout_regression
from srl.profiler.schema import (
    HardwareDescriptor,
    ModelDescriptor,
    Parallelism,
    ProfileRecord,
    RolloutPayload,
    RunDescriptor,
    SoftwareDescriptor,
)


def _descriptors() -> tuple[ModelDescriptor, HardwareDescriptor, SoftwareDescriptor]:
    return (
        ModelDescriptor(
            name="synthetic-qwen",
            hidden_size=4096,
            num_layers=36,
            num_attention_heads=32,
            num_query_groups=8,
            extra={"intermediate_size": 11008},
        ),
        HardwareDescriptor(gpu_name="H100", gpu_count=1),
        SoftwareDescriptor(backend="vllm", backend_version="0.26.0"),
    )


def _synthetic_records(model) -> list[ProfileRecord]:
    model_descriptor, hardware, software = _descriptors()
    parallelism = Parallelism()
    records: list[ProfileRecord] = []
    for phase in ("prefill", "decode"):
        for repeat in range(4):
            for request_count in (1, 2, 4, 8, 16, 32):
                for length in (128, 256, 512, 1024):
                    if phase == "prefill":
                        n_tokens = request_count * length
                        sum_l2 = request_count * length**2
                        ctxsum = 0
                        input_tokens = n_tokens
                    else:
                        n_tokens = request_count
                        sum_l2 = 0
                        ctxsum = request_count * length
                        input_tokens = 0
                    workload = RolloutWorkload(
                        phase=phase,
                        n_tokens=n_tokens,
                        sum_l2=sum_l2,
                        ctxsum=ctxsum,
                        request_count=request_count,
                    )
                    payload = RolloutPayload(
                        phase=phase,
                        request_count=request_count,
                        input_tokens=input_tokens,
                        output_tokens=request_count,
                        n_tokens=n_tokens,
                        sum_l=float(request_count * length),
                        sum_l2=float(sum_l2),
                        ctxsum=float(ctxsum),
                        latency_s=float(model.predict(workload)),
                    )
                    records.append(
                        ProfileRecord(
                            record_type="rollout",
                            run=RunDescriptor(
                                run_id=f"{phase}-{repeat}",
                                repeat=repeat,
                            ),
                            model=model_descriptor,
                            hardware=hardware,
                            software=software,
                            parallelism=parallelism,
                            payload=payload,
                        )
                    )
    return records


def test_roofline_fit_and_artifact_roundtrip(tmp_path: Path):
    source = UnifiedRoofline(
        F=0.002,
        W=0.0005,
        G=2e-7,
        A_p=3e-9,
        A_d=4e-9,
        b=16,
    )
    artifact = UnifiedRoofline.fit(_synthetic_records(source))
    path = artifact.save(tmp_path / "roofline.json")
    loaded_artifact = CostArtifact.load(path)
    loaded = UnifiedRoofline.from_artifact(loaded_artifact, Parallelism())
    query = RolloutWorkload(
        phase="mixed",
        n_tokens=64,
        sum_l2=32768,
        ctxsum=8192,
        request_count=8,
    )
    assert loaded.predict(query) == pytest.approx(source.predict(query), rel=1e-3)
    assert artifact.entries[0].validation_metrics.r_squared > 0.999
    assert artifact.entries[0].parallelism == Parallelism()


def test_zero_validation_fraction_uses_every_sample_for_training():
    source = UnifiedRoofline(F=0.002, W=0.0005, G=2e-7, A_p=3e-9, A_d=4e-9)
    records = _synthetic_records(source)

    artifact = UnifiedRoofline.fit(records, validation_fraction=0)
    entry = artifact.entries[0]

    assert entry.train_metrics.n_samples == len(records)
    assert entry.validation_metrics.n_samples == 0


def test_psrl_phase_fit_and_closed_form():
    source = PSRLFitted(
        prefill_attn_b=0.0002,
        prefill_attn_k=2e-9,
        prefill_other_floor=0.001,
        prefill_other_b=0.0002,
        prefill_other_k=3e-7,
        decode_attn_b=0.0003,
        decode_attn_k=4e-9,
        decode_other_floor=0.0012,
        decode_other_b=0.0001,
        decode_other_k=2e-5,
    )
    artifact = PSRLFitted.fit(_synthetic_records(source))
    loaded = PSRLFitted.from_artifact(artifact, Parallelism())
    workload = RolloutWorkload(
        phase="decode",
        n_tokens=8,
        sum_l2=0,
        ctxsum=8192,
        request_count=8,
    )
    assert loaded.predict(workload) == pytest.approx(source.predict(workload), rel=1e-3)
    floor, alpha, beta = loaded.decode_coeffs(8, 8192)
    assert floor == 0
    assert alpha > 0
    assert beta >= 0
    assert artifact.entries[0].validation_metrics.r_squared > 0.999


def test_distserve_fit_records_canonical_identifiability():
    source = DistServe(
        h=4096,
        m=11008,
        C1=1e-15,
        C2=2e-13,
        C3=0.001,
        C4=0,
        C5=3e-13,
        b=16,
        F=0.002,
    )
    artifact = DistServe.fit(_synthetic_records(source))
    entry = artifact.entries[0]
    assert entry.diagnostics["c3_c4_identifiability"] == "single-model-canonical-minimum-norm"
    loaded = DistServe.from_artifact(artifact, Parallelism())
    query = RolloutWorkload(
        phase="prefill",
        n_tokens=1024,
        sum_l2=1024**2,
        ctxsum=0,
        request_count=1,
    )
    assert loaded.predict(query) == pytest.approx(source.predict(query), rel=1e-3)


@pytest.mark.parametrize(
    ("source", "fit"),
    [
        (
            UnifiedRoofline(F=0.002, W=0.0005, G=2e-7, A_p=3e-9, A_d=4e-9),
            UnifiedRoofline.fit,
        ),
        (
            PSRLFitted(
                prefill_attn_b=0.0002,
                prefill_attn_k=2e-9,
                prefill_other_floor=0.001,
                prefill_other_b=0.0002,
                prefill_other_k=3e-7,
                decode_attn_b=0.0003,
                decode_attn_k=4e-9,
                decode_other_floor=0.0012,
                decode_other_b=0.0001,
                decode_other_k=2e-5,
            ),
            PSRLFitted.fit,
        ),
        (
            DistServe(
                h=4096,
                m=11008,
                C1=1e-15,
                C2=2e-13,
                C3=0.001,
                C4=0,
                C5=3e-13,
                F=0.002,
            ),
            DistServe.fit,
        ),
    ],
    ids=("roofline", "psrl", "distserve"),
)
def test_vllm_regression_plot_renders_all_model_families(tmp_path: Path, source, fit):
    records = _synthetic_records(source)
    artifact = fit(records)

    paths = plot_vllm_rollout_regression(records, artifact, tmp_path / "plots")

    assert paths == [tmp_path / "plots" / f"{artifact.model_type}_tp1_pp1_dp1_cp1.png"]
    assert paths[0].stat().st_size > 0


def test_vllm_regression_plot_names_each_topology(tmp_path: Path):
    source = UnifiedRoofline(F=0.002, W=0.0005, G=2e-7, A_p=3e-9, A_d=4e-9)
    tp1_records = _synthetic_records(source)
    tp2_records = [
        replace(
            record,
            hardware=replace(record.hardware, gpu_count=2),
            parallelism=Parallelism(tp=2),
        )
        for record in tp1_records
    ]
    records = tp1_records + tp2_records
    artifact = UnifiedRoofline.fit(records)

    paths = plot_vllm_rollout_regression(records, artifact, tmp_path / "plots")

    assert {path.name for path in paths} == {
        "roofline_tp1_pp1_dp1_cp1.png",
        "roofline_tp2_pp1_dp1_cp1.png",
    }
    assert all(path.stat().st_size > 0 for path in paths)


def test_artifact_lookup_rejects_unprofiled_topology():
    source = UnifiedRoofline(F=0.002, W=0.001, G=1e-7, A_p=1e-9, A_d=1e-9)
    artifact = UnifiedRoofline.fit(_synthetic_records(source))
    with pytest.raises(KeyError, match="exactly one"):
        UnifiedRoofline.from_artifact(
            artifact,
            Parallelism(tp=2, pp=1, dp=1, cp=1),
        )
