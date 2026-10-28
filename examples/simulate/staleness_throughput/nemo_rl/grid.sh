# NeMo-RL: `sampler` must be pinned to windowed. Under the default in_order,
# horizon reads max_lookahead_versions instead and the L2 cancel never runs, so
# sweeping max_staleness_versions would move nothing.
# 0 is excluded: the windowed gate never opens at that value, livelocking the run.
#
# max_inflight_prompts and max_buffered_rollouts are pinned to scale with
# BATCH_SIZE rather than left at their class defaults (32, 64):
# NeMoRLPolicyConfig.ingress_window_groups shows the effective admission
# window is min(max_inflight_prompts, max_buffered_rollouts,
# max_staleness_versions * BATCH_SIZE), so at the old BATCH_SIZE=8 the
# defaults gave headroom for most of this axis (32 = 4x, binding only at
# staleness>=5), but at BATCH_SIZE=32 the same 32/64 are at or below 1x/2x
# batch size and cap every point on this axis at staleness 0 regardless of
# the knob -- a silent admission-never-opens trap (see the sweep README's
# own warning about this), not a real result. *6 covers the whole swept
# range (max 5) with the same margin nemo_rl_in_order/grid.sh uses for its
# own buffer.
PRESET=nemo_rl
PRIMARY_KNOB=policy.max_staleness_versions
PRIMARY_VALUES=(1 2 3 4 5)
PRIMARY_LABEL="nemo_rl max_staleness_versions"
PINS=(policy.sampler=windowed "policy.max_inflight_prompts=$((BATCH_SIZE * 6))" "policy.max_buffered_rollouts=$((BATCH_SIZE * 6))")

# SECONDARY_KNOB=policy.sample_freshest_first
# SECONDARY_VALUES=(false true)
