# OpenRLHF: a permit pool counted in whole batch lumps, not per-sample age.
# partial_rollout_enable must be on or the policy asserts for any queue size
# above 1. Turning it on also makes the pause mode in_place, which is the same
# switch in the real system rather than a separate choice.
PRESET=openrlhf
PRIMARY_KNOB=policy.async_queue_size
PRIMARY_VALUES=(1 2 3 4 5)
PRIMARY_LABEL="openrlhf async_queue_size"
PINS=(policy.partial_rollout_enable=true)
