# prime-rl: `max_off_policy_steps` is one lower bound shared by the L2 cancel and
# the L3 sweep, so a change moves both at once.
PRESET=prime_rl
PRIMARY_KNOB=policy.max_off_policy_steps
PRIMARY_VALUES=(0 1 2 3 4 5)
PRIMARY_LABEL="prime_rl max_off_policy_steps (versions)"
PINS=()
