# psrl: a Reserve/Occupy/Consume inventory whose window is `staleness`.
#
# The knob reaches the policy through SimConfig.staleness_horizon rather than a
# constructor argument, and the same value widens the ingress window, so setting
# the preset field is all that is needed.
PRESET=psrl
PRIMARY_KNOB=policy.staleness
PRIMARY_VALUES=(0 1 2 3 4 5)
PRIMARY_LABEL="psrl staleness (versions)"
PINS=()

# SECONDARY_KNOB=policy.proactive_filter
# SECONDARY_VALUES=(false true)
