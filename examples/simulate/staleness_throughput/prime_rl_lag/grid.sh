# prime-rl: `target_lag` gates how far dispatch can run ahead of training,
# independent of `max_off_policy_steps`. At the default target_lag=1 (see
# prime_rl/grid.sh), the regime never lets staleness develop at all, masking
# whether max_off_policy_steps's bound actually holds once it has something
# to catch -- and whether the completion-time check closes the same
# publish-only-sweep race TRL and verl_v1's drop strategy had. Paneled
# against two off-policy-steps settings to see both where staleness starts
# developing and whether the bound then holds.
PRESET=prime_rl
PRIMARY_KNOB=policy.target_lag
PRIMARY_VALUES=(1 2 3 4 5)
PRIMARY_LABEL="prime_rl target_lag"
SECONDARY_KNOB=policy.max_off_policy_steps
SECONDARY_VALUES=(1 3)
PINS=()
