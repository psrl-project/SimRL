# verl legacy: `staleness_threshold` is a sample-overage RATIO, not a version
# count. It admits at most int(B * (1 + threshold)) groups between syncs, so the
# axis label must not claim versions even though it shares the 0..5 range.
PRESET=verl
PRIMARY_KNOB=policy.staleness_threshold
PRIMARY_VALUES=(0 1 2 3 4 5)
PRIMARY_LABEL="verl staleness_threshold (sample-overage ratio, not versions)"
PINS=()

# SECONDARY_KNOB=policy.require_batches
# SECONDARY_VALUES=(1 2)
