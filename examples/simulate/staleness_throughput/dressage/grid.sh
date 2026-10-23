# Dressage: `keep_versions` counts distinct observed versions, keyed on each
# group's last trainable token. Zero means the filter is OFF, which is the loose
# end of the range rather than the tight one, so this axis starts at 1.
PRESET=dressage
PRIMARY_KNOB=policy.keep_versions
PRIMARY_VALUES=(1 2 3 4 5)
PRIMARY_LABEL="dressage keep_versions (distinct versions kept)"
PINS=()

# SECONDARY_KNOB=policy.max_partial_rollout_preempts
# SECONDARY_VALUES=(1 2 4)
