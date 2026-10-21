# Experiments

Generated, not curated. `smoke.py` is the template and the only tracked file here.

## Writing one

1. Copy `smoke.py` to `<your-experiment>.py`.
2. Set `POINT` (or a list of points) to the workload you are testing on.
3. Set `ARMS` to the controls and treatments. The control must be the configuration the treatment
   is supposed to beat, not the configuration it was tuned against.
4. Run it. It prints one gate verdict per arm and writes `outputs/simulate/validate_smoke/summary.json`.
5. Read `../reference/gates.md` before quoting any number it prints.

## What an experiment may not do

- Compute a performance number without running the simulator.
- Compare against a result file produced in another session or with another binary.
- Keep a point that failed a gate in the table or in the mean.
