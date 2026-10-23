# slime: no version comparison anywhere. The maximum staleness is identically
# `update_weights_interval`, which is a loop interval, so 0 is not a setting.
PRESET=slime
PRIMARY_KNOB=policy.update_weights_interval
PRIMARY_VALUES=(1 2 3 4 5)
PRIMARY_LABEL="slime update_weights_interval (training steps per sync)"
PINS=()

# SECONDARY_KNOB=policy.over_sampling_ratio
# SECONDARY_VALUES=(1.0 1.5)
