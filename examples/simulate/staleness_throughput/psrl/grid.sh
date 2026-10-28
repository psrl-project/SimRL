# psrl: a Reserve/Occupy/Consume inventory whose window is `staleness`.
#
# The knob reaches the policy through SimConfig.staleness_horizon rather than a
# constructor argument, and the same value widens the ingress window, so setting
# the preset field is all that is needed.
PRESET=psrl
PRIMARY_KNOB=policy.staleness
PRIMARY_VALUES=(0 1 2 3 4 5)
PRIMARY_LABEL="psrl staleness"
# Pinned to 40: capping per-instance rollout concurrency (psrl_aligned.sh's
# own ACTIVE_CAP knob, default 0 = uncapped) raised throughput 14-20% at
# staleness>=3 with no change to the achieved max staleness, so 40 replaces
# the uncapped default here rather than staying a side-by-side comparison.
PINS=(rollout.router.max_concurrent_seqs_per_instance=40)

# SECONDARY_KNOB=policy.proactive_filter
# SECONDARY_VALUES=(false true)
