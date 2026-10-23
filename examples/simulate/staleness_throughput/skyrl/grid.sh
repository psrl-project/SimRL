# SkyRL: `max_staleness_steps` drives the admission quota and a violation
# counter and never discards, so this axis is expected to move span without
# bounding the composition. A flat figure here is the measurement.
PRESET=skyrl
PRIMARY_KNOB=policy.max_staleness_steps
PRIMARY_VALUES=(0 1 2 3 4 5)
PRIMARY_LABEL="skyrl max_staleness_steps (versions, warn-only)"
PINS=()
