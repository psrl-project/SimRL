# Reference

Lookup material for the installed interface: console scripts, configuration keys, on-disk
formats, the validation method and the figures that can be re-derived, and the limits that hold
today. Every page was written from the code, so when a page and the code disagree, the code wins.

| Page | Contents |
|---|---|
| {doc}`cli` | The seven console scripts, their subcommands and flags, and the environment variables the code reads |
| {doc}`configuration` | The `srl` Hydra app: config groups, every key, defaults, and how a preset becomes a policy |
| {doc}`schemas` | `ProfileRecord`, `CostArtifact`, and the telemetry streams |
| {doc}`validation` | How accuracy is measured, the gates, and the figures that reproduce from local data or pinned tests |
| {doc}`limitations` | What the code does not do today |

Accuracy figures live on {doc}`validation` and nowhere else in the reference. Each one carries a
sample count and says whether the samples were held out, because that decides what the number can
be used for.

```{toctree}
:maxdepth: 2

cli
configuration
schemas
validation
limitations
```
