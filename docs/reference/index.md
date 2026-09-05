# Reference

Lookup material: exact commands, exact configuration keys, exact on-disk formats, and an
honest account of what has and has not been validated.

| Page | Contents |
|---|---|
| {doc}`cli` | Every subcommand and flag across the five entry points |
| {doc}`configuration` | Every Hydra config group and key, with defaults |
| {doc}`schemas` | `ProfileRecord`, `CostArtifact`, and the telemetry streams |
| {doc}`validation` | Every measured accuracy figure, with sample counts and caveats |
| {doc}`limitations` | What is known not to work, and deferred work with acceptance criteria |

If you are looking for a number to quote, {doc}`validation` is the page that owns it. Every
figure there carries its sample count and whether it was held out, which is usually the part
that decides whether the number means what you want it to mean.

```{toctree}
:maxdepth: 2

cli
configuration
schemas
validation
limitations
```
