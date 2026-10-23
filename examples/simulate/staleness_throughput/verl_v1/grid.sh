# verl v1: the L1 feed and the L3 bound, swept together, because the bound can
# only bind once the feed lets rollout run ahead of training. Both assert
# strictly positive, so neither axis starts at 0.
PRESET=verl_v1
PRIMARY_KNOB=policy.num_warmup_batches
PRIMARY_VALUES=(1 2 3 4 5)
PRIMARY_LABEL="verl_v1 num_warmup_batches (batches prefetched)"
SECONDARY_KNOB=policy.max_off_policy_threshold
SECONDARY_VALUES=(1 2 3 4 5)
PINS=()

# SECONDARY_KNOB=policy.max_off_policy_strategy
# SECONDARY_VALUES=(drop wait)
