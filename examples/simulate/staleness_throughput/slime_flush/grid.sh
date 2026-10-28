# slime, sweep (b): the flush cadence is a KV-correctness knob, not a staleness
# bound. 0 never aborts after the first publication, 1 aborts at every
# publication, and N aborts every N-th one. Aborted flat samples keep their
# prefix and are requeued, so the composition should barely move while the
# re-prefill cost changes the bar height.
#
# Fixed at the middle of sweep (a)'s L1 axis, and 32 versions so the staleness
# levels are not the run's own length.
PRESET=slime
PRIMARY_KNOB=policy.flush_cache_interval
PRIMARY_VALUES=(0 1 2 4 8)
PRIMARY_LABEL="slime flush_cache_interval (0 = never abort)"
# server_concurrency scales the same way slime_l1/grid.sh's own copy does
# (see its comment): a formula referencing BATCH_SIZE/GROUP_SIZE/N_INSTANCES
# rather than a constant calibrated against one past BATCH_SIZE, so
# threshold_groups() keeps clearing slime.validate()'s own admission
# assertion at any BATCH_SIZE. group_prefetch_factor stays pinned at 2 --
# it is "the middle of sweep (a)'s L1 axis" by construction regardless of
# BATCH_SIZE, not a value tied to the old regime.
PINS=("policy.server_concurrency=$((BATCH_SIZE * GROUP_SIZE / N_INSTANCES))" "policy.group_prefetch_factor=2")
export N_VERSIONS=32
