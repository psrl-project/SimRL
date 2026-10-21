# Experiments

Generated drivers, gitignored except this file. Each exposes
`build(spec: dict, output_dir: Path | None) -> tuple[Simulator, int]` so `scripts/replay.py` can
re-run one row with full telemetry. See `workflow/validate/experiments/smoke.py` for the shape.
