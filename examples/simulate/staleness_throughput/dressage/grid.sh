# Dressage: `keep_versions` counts distinct observed versions, keyed on each
# group's last trainable token. Zero means the filter is OFF, which is the loose
# end of the range rather than the tight one, so this axis starts above it.
# keep_versions=1 used to undershoot 3072 consumed trajectories (2560) within
# n_versions=12: the old `peek_batch`-only passive filter excluded enough of
# the ready pool with no active eviction to trigger a dataset replacement.
# `on_batch_attempt` now actively evicts (with REPLACE backfill), matching
# real Dressage's `filter_pending`, and keep_versions=1 reaches 3072 again.
PRESET=dressage
PRIMARY_KNOB=policy.keep_versions
PRIMARY_VALUES=(1 2 3 4 5)
PRIMARY_LABEL="dressage keep_versions"
PINS=()

# SECONDARY_KNOB=policy.max_partial_rollout_preempts
# SECONDARY_VALUES=(1 2 4)
