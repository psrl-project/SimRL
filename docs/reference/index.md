# Reference

Lookup material: exact commands, exact configuration keys, exact on-disk formats, and an
honest account of what has and has not been validated.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`terminal;1.5em` CLI
:link: cli
:link-type: doc

Every subcommand and flag across the five entry points.
:::

:::{grid-item-card} {octicon}`sliders;1.5em` Configuration
:link: configuration
:link-type: doc

Every Hydra config group and key, with defaults.
:::

:::{grid-item-card} {octicon}`file-code;1.5em` Schemas
:link: schemas
:link-type: doc

`ProfileRecord`, `CostArtifact`, and the telemetry streams.
:::

:::{grid-item-card} {octicon}`beaker;1.5em` Validation
:link: validation
:link-type: doc

Every measured accuracy figure, with sample counts and caveats.
:::

::::

{doc}`limitations` lists what is known not to work, and the deferred work items with
their acceptance criteria.

```{toctree}
:maxdepth: 2

cli
configuration
schemas
validation
limitations
```
