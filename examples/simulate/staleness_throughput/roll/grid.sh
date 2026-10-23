# ROLL: `async_generation_ratio` is a ratio, but horizon takes its ceiling, so it
# acts as the version window for both the admission budget and the L2 GC.
PRESET=roll
PRIMARY_KNOB=policy.async_generation_ratio
PRIMARY_VALUES=(0 1 2 3 4 5)
PRIMARY_LABEL="roll async_generation_ratio (ceiling is the version window)"
PINS=()
