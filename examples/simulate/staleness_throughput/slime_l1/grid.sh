# slime, sweep (a): the L1 group threshold is the only knob that moves a
# trajectory's age. `server_concurrency` is fixed per engine and the prefetch
# factor scales the group pool on top of it, which is the product the worker
# actually gates admission on.
#
# The engine's sample concurrency is left at the shared regime's value. What
# bounds concurrently running samples here is the KV pool, not the prefetch
# factor, so extra groups queue instead of generating, which is what the
# per-sample semaphore does in the worker.
#
# 32 versions so the top of the axis is not the run's own length.
PRESET=slime
PRIMARY_KNOB=policy.group_prefetch_factor
PRIMARY_VALUES=(1 2 4 8)
PRIMARY_LABEL="slime group_prefetch_factor"
# server_concurrency is a formula, not a bare constant: threshold_groups()
# computes per_engine = max(1, server_concurrency * N_INSTANCES // GROUP_SIZE),
# so this choice of server_concurrency makes per_engine land exactly on
# BATCH_SIZE regardless of the shared regime's own values, instead of a
# number calibrated against one past BATCH_SIZE that silently stops clearing
# `slime.validate()`'s own `threshold >= batch_size` assertion at another.
PINS=("policy.server_concurrency=$((BATCH_SIZE * GROUP_SIZE / N_INSTANCES))" "policy.flush_cache_interval=1")
export N_VERSIONS=32

# Threshold formula: max(1, server_concurrency * N_INSTANCES // GROUP_SIZE) * factor
# = BATCH_SIZE * factor groups, so the axis runs BATCH_SIZE, 2x, 4x, 8x groups
# against a batch of BATCH_SIZE. The first point is the minimum the run can
# admit, the second is where the sample slots fill, and the rest
# over-subscribe the KV pool.
