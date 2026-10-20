"""Load fitted rollout and training entries for an auto-search candidate."""

from __future__ import annotations

import argparse
import logging

from srl.cost.rollout import PSRLFitted, RolloutWorkload
from srl.cost.training import (
    HydraulisTrainingCost,
    TrainingParallelism,
    TrainingWorkload,
)
from srl.profiler.schema import Parallelism


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-artifact", required=True)
    parser.add_argument("--training-artifact", required=True)
    args = parser.parse_args()

    rollout = PSRLFitted.from_artifact(
        args.rollout_artifact,
        Parallelism(tp=1, pp=1, dp=1, cp=1),
    )
    decode_latency = rollout.predict(
        RolloutWorkload(
            phase="decode",
            n_tokens=8,
            sum_l2=0,
            ctxsum=8192,
            request_count=8,
        )
    )

    training = HydraulisTrainingCost.from_artifact(args.training_artifact)
    estimate = training.estimate(
        TrainingWorkload(
            sequence_lengths=(1024,) * 8,
            micro_batch_size=1,
        ),
        TrainingParallelism(tp=1, pp=1, dp=1, cp=1),
    )
    logging.info("Decode latency: %.6f seconds.", decode_latency)
    logging.info(
        "Training latency: %.6f seconds, peak memory: %d bytes.",
        estimate.latency_s,
        estimate.peak_memory_bytes,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
