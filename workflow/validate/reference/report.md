# Report template

A report is complete only when every section is present. Numbers come from gated runs only.

## 1. Question and change

One sentence on what was tested, the flag that enables it, and the control it is compared to.

## 2. Session

- git head and the uncommitted `srl/` files, from `scripts/preflight.py`.
- Binary md5 before and after the sweep.
- `SIMRL_KV_CACHE` value, and any preflight warning.

## 3. Gated results

| point | control span | treatment span | speedup | consumed_groups | tokens_per_group drift | treatment concurrency | best static concurrency |
|---|---|---|---|---|---|---|---|

- Geometric mean over gated points, with the point count.
- Tuned points and untuned points in separate tables. The headline is the untuned table.
- State that runs are deterministic and that no seed variance was measured, unless it was.

## 4. Failed gates

One line per dropped point: point, gate, observed value, threshold. Empty only when nothing failed.

## 5. Not established

What this experiment cannot support, for example long-context accuracy, other TP, other models.
