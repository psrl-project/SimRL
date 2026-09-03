"""GPU spec loading, dtype canonicalization, and name slugging."""

from __future__ import annotations

import pytest

from dryrun.cost.kernel.hardware import (
    GpuSpec,
    canonical_dtype,
    dtype_bytes,
    gpu_slug,
    load_gpu_spec,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("NVIDIA H20", "h20"),
        ("h20", "h20"),
        ("NVIDIA H100 80GB HBM3", "h100_80gb_hbm3"),
        ("NVIDIA H200 SXM", "h200_sxm"),
    ],
)
def test_gpu_slug_normalizes_device_names(raw, expected):
    assert gpu_slug(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("bf16", "bfloat16"), ("bfloat16", "bfloat16"), ("fp16", "float16"), ("fp8", "fp8")],
)
def test_canonical_dtype_folds_aliases(raw, expected):
    assert canonical_dtype(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("bfloat16", 2), ("bf16", 2), ("float16", 2), ("float32", 4), ("fp8", 1)],
)
def test_dtype_bytes(raw, expected):
    assert dtype_bytes(raw) == expected


def test_unknown_dtype_is_rejected():
    with pytest.raises(ValueError):
        dtype_bytes("int3")


def test_bundled_h20_spec_matches_measured_hardware():
    spec = load_gpu_spec("h20")
    assert spec.slug == "h20"
    assert spec.sm_count == 78
    assert spec.memory_bandwidth_bytes_per_s == pytest.approx(4.0e12)
    assert spec.flops_for("bfloat16") == pytest.approx(148e12)
    assert spec.flops_for("bf16") == pytest.approx(148e12)


def test_missing_dtype_flops_is_rejected():
    spec = load_gpu_spec("h20")
    with pytest.raises(KeyError):
        spec.flops_for("int4")


def test_unrecognized_dtype_message_names_recognized_dtypes():
    """An entirely unrecognized dtype should say the dtype is not recognized."""
    spec = load_gpu_spec("h20")
    with pytest.raises(KeyError, match=r"not recognized\. Recognized dtype names:"):
        spec.flops_for("int4")


def test_recognized_but_missing_dtype_message_names_gpu_dtypes():
    """A recognized dtype with no published figure should list the GPU's available dtypes."""
    spec = load_gpu_spec("h20")
    with pytest.raises(KeyError, match=r"has no 'int8' throughput\. Available:"):
        spec.flops_for("int8")


def test_unknown_gpu_is_rejected():
    with pytest.raises(KeyError):
        load_gpu_spec("gtx770")


def test_custom_spec_path_is_honored(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(
        "toaster:\n"
        "  name: Kitchen Toaster\n"
        "  sm_count: 1\n"
        "  memory_bytes: 1024\n"
        "  memory_bandwidth_bytes_per_s: 2.0\n"
        "  link_bandwidth_bytes_per_s: 1.0\n"
        "  tc_flops_per_s:\n"
        "    bfloat16: 3.0\n",
        encoding="utf-8",
    )
    spec = load_gpu_spec("toaster", path=path)
    assert isinstance(spec, GpuSpec)
    assert spec.name == "Kitchen Toaster"
    assert spec.flops_for("bfloat16") == pytest.approx(3.0)
