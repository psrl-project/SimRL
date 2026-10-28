# TRL: the most direct bound in the study. `max_staleness` compares the
# generating version against the training version at dequeue.
PRESET=trl
PRIMARY_KNOB=policy.max_staleness
PRIMARY_VALUES=(0 1 2 3 4 5)
PRIMARY_LABEL="trl max_staleness"
PINS=()
