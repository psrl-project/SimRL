# Contributing to SimRL

Thanks for considering a contribution. Bug reports, fixes, new cost models, new policy presets and
documentation improvements are all welcome.

## Getting set up

Follow [Install](README.md#install) to set up the environment, then verify it:

```bash
pytest                     # tests needing MODEL_PATH are skipped without it
ruff check .
make -C docs strict        # warnings promoted to errors
```

## Where things live

[`docs/development/index.md`](docs/development/index.md) lists SimRL's six extension points (kernel
op, model family, profiler adapter, rollout cost model, training cost model, staleness policy), each
a small protocol with one registry, and the structural rules that keep the cost path free of `torch`
and keep a registry's population path single.

[`docs/development/testing.md`](docs/development/testing.md) covers what each test group needs, what
a skip means, and the pinned accuracy figures. [`docs/development/docs.md`](docs/development/docs.md)
covers building the documentation site and the conventions that fail its strict build, including the
rule that a fact is stated once, on the page that owns it.

Read `.claude/coding-style.md` before writing code. Two of its rules prevent silent bugs: an
identifier string never gains a period or a case change, and a comment says what the code cannot.

## Adding a policy preset or a routing strategy

[`docs/simulation/policies.md`](docs/simulation/policies.md) has a "Writing a policy" section with
the exact steps: subclass `StalenessPolicy`, implement `admit_quota`/`peek_batch`, register the class
in `POLICY_REGISTRY`, and add a Hydra preset under `srl/config/defaults/policy/`.
[`docs/simulation/routing.md`](docs/simulation/routing.md) has the equivalent for a routing strategy:
one module under `srl/component/rollout/router/strategies/`, a class decorated with
`@register_strategy("name")`, and an import in that package's `__init__.py`. Neither requires
touching the simulator.

## Pull requests

- Run the three commands under "Getting set up" before opening a PR. `make -C docs strict` is what
  catches a broken cross-reference or an orphaned page.
- Changing a console flag, a config key or a measured number means changing the page that owns that
  fact in the same change. The ownership table is in `docs/development/docs.md`.
- Keep commits focused. Prefer a few small PRs over one large one.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE).
