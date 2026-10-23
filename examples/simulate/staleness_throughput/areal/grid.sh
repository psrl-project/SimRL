# AReaL: a cumulative token bucket whose width is `max_head_offpolicyness`,
# reaching the policy through SimConfig.staleness_horizon.
PRESET=areal
PRIMARY_KNOB=policy.max_head_offpolicyness
PRIMARY_VALUES=(0 1 2 3 4 5)
PRIMARY_LABEL="areal max_head_offpolicyness (versions)"
PINS=()
