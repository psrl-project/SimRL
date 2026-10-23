# SkyRL: `max_staleness_steps` drives an admission quota and a violation
# counter. Nothing is ever discarded, so `maxS` is never pulled back.
PRESET=skyrl
PRIMARY_KNOB=policy.max_staleness_steps
PRIMARY_VALUES=(0 1 2 3 4 5)
PRIMARY_LABEL="skyrl max_staleness_steps (versions, warn-only)"
PINS=()
