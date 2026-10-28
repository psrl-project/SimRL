# verl_v1: the wait strategy, mirroring the drop strategy's own grid
# (num_warmup_batches x max_off_policy_threshold, see verl_v1/grid.sh) but
# with max_off_policy_strategy=wait pinned. Real verl's wait never re-filters
# a group once it has finished generating, however stale it later becomes --
# this exists to confirm the simulator reproduces that absence, not to find
# a bound.
PRESET=verl_v1
PRIMARY_KNOB=policy.num_warmup_batches
PRIMARY_VALUES=(1 2 3 4 5)
PRIMARY_LABEL="verl_v1 num_warmup_batches (wait)"
SECONDARY_KNOB=policy.max_off_policy_threshold
SECONDARY_VALUES=(1 2 3 4 5)
PINS=(policy.max_off_policy_strategy=wait)
