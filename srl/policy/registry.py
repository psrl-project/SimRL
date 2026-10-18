"""Maps a preset name to its policy class."""

from __future__ import annotations

from .areal import ArealPolicy
from .openrlhf import OpenRLHFPolicy
from .psrl import PSRLPolicy
from .roll import RollPolicy
from .skyrl import SkyRLPolicy
from .slime import SlimePolicy
from .verl import VerlPolicy
from .verl_v1 import VerlV1Policy

POLICY_REGISTRY: dict[str, type] = {
    "psrl": PSRLPolicy,
    "areal": ArealPolicy,
    "roll": RollPolicy,
    "slime": SlimePolicy,
    "verl": VerlPolicy,
    "verl_v1": VerlV1Policy,
    "skyrl": SkyRLPolicy,
    "openrlhf": OpenRLHFPolicy,
}
