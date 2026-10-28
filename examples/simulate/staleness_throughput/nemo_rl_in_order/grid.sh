# NeMo-RL: the in_order sampler, exact-match admission reserved per step.
# Uses max_lookahead_versions instead of max_staleness_versions -- windowed's
# own knob does nothing here (see nemo_rl/grid.sh's comment on this same
# swap). 0 is excluded on the same pattern as windowed's own floor:
# dispatch_index starts equal to train_version, so a zero lookahead blocks
# admission from the first tick, before anything has a chance to run.
#
# max_buffered_rollouts is pinned to cover the whole PRIMARY_VALUES range
# (6 = max(PRIMARY_VALUES) + 1), not left at its own default (64). The real
# source's own `InOrderSampler.required_buffer_capacity` requires
# `groups_per_step * (lookahead + 1)`, and its own validator rejects a config
# that under-provisions it -- left at a fixed 64, lookahead >= 3 silently
# plateaus at lookahead=2's own achieved staleness (the buffer, not the
# gate, then bounds it), reproducing a configuration the real system would
# refuse to start rather than one it actually runs.
#
# max_inflight_prompts is pinned the same way, correcting an earlier claim
# here that it needed no equivalent fix: it is not tied to *lookahead*
# specifically in the real source, but `NeMoRLPolicyConfig.ingress_window_groups`
# shows the effective admission window is min(max_inflight_prompts,
# max_buffered_rollouts, lookahead * BATCH_SIZE) regardless of sampler, so
# it is tied to BATCH_SIZE generally. Left at its class default (32) it
# equals exactly 1x the new BATCH_SIZE=32 and becomes the binding term at
# every point on this axis, reproducing the same silent admission-trap flat
# figure nemo_rl/grid.sh's own comment describes, independent of lookahead.
PRESET=nemo_rl
PRIMARY_KNOB=policy.max_lookahead_versions
PRIMARY_VALUES=(1 2 3 4 5)
PRIMARY_LABEL="nemo_rl max_lookahead_versions"
PINS=(policy.sampler=in_order "policy.max_buffered_rollouts=$((BATCH_SIZE * 6))" "policy.max_inflight_prompts=$((BATCH_SIZE * 6))")
