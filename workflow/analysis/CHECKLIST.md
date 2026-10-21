# Analysis checklist

Tick in order. A box that cannot be ticked is a stop condition, not a note.

## Setup

- [ ] `SIMRL_WORKSPACE` exported and `scripts/preflight.py` exits 0
- [ ] `env/simrl.sh` sourced
- [ ] The input `summary.json` exists and every row carries `row_id` and `replay`
- [ ] Row selection recorded, with one reason per row, at most 6 rows
- [ ] Traces go to a local disk, not the shared filesystem

## Per row

- [ ] Replay completed without livelock
- [ ] Fidelity: every gated field matches the row, floats to 1e-9 relative and counts exactly
- [ ] Every cross-source check passed
- [ ] `diagnosis.json` and all four figures written under `outputs/analysis/`

## Reporting

- [ ] Every bound carries its assumptions
- [ ] Idle breakdown shares sum to within 1% of 1.0, or the residual is named
- [ ] Hypotheses ranked by share of total idle time
- [ ] Each hypothesis names the `validate` experiment that would falsify it
- [ ] No speedup claimed anywhere
- [ ] Unexplained waits reported with their count and total seconds
