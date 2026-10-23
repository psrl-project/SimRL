# NeMo-RL: `sampler` must be pinned to windowed. Under the default in_order,
# horizon reads max_lookahead_versions instead and the L2 cancel never runs, so
# sweeping max_staleness_versions would move nothing.
PRESET=nemo_rl
PRIMARY_KNOB=policy.max_staleness_versions
PRIMARY_VALUES=(0 1 2 3 4 5)
PRIMARY_LABEL="nemo_rl max_staleness_versions (versions, sampler=windowed)"
PINS=(policy.sampler=windowed)

# SECONDARY_KNOB=policy.sample_freshest_first
# SECONDARY_VALUES=(false true)
