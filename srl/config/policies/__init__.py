"""One typed preset per asynchronous-RL system."""

from __future__ import annotations

from dataclasses import fields

from .areal import ArealPolicyConfig
from .dressage import DressagePolicyConfig
from .nemo_rl import NeMoRLPolicyConfig
from .openrlhf import OpenRLHFPolicyConfig
from .psrl import PsrlPolicyConfig
from .roll import RollPolicyConfig
from .skyrl import SkyRLPolicyConfig
from .slime import SlimePolicyConfig
from .trl import TRLPolicyConfig
from .verl import VerlPolicyConfig
from .verl_v1 import VerlV1PolicyConfig

PRESETS: dict[str, type] = {
    "psrl": PsrlPolicyConfig,
    "areal": ArealPolicyConfig,
    "roll": RollPolicyConfig,
    "slime": SlimePolicyConfig,
    "verl": VerlPolicyConfig,
    "verl_v1": VerlV1PolicyConfig,
    "skyrl": SkyRLPolicyConfig,
    "openrlhf": OpenRLHFPolicyConfig,
    "dressage": DressagePolicyConfig,
    "nemo_rl": NeMoRLPolicyConfig,
    "trl": TRLPolicyConfig,
}


def preset_config(cfg) -> object:
    """Return the typed preset for a composed `policy:` node.

    Hydra composes the selected group into a plain dict, which drops the
    dataclass methods `build`, `horizon` and `ingress_window_groups`.
    """
    name = cfg.get("name", "psrl")
    cls = PRESETS.get(name)
    if cls is None:
        raise ValueError(f"Unknown policy: {name!r}.")
    known = {f.name for f in fields(cls)}
    return cls(**{key: value for key, value in cfg.items() if key in known})


__all__ = [
    "PRESETS",
    "ArealPolicyConfig",
    "DressagePolicyConfig",
    "NeMoRLPolicyConfig",
    "OpenRLHFPolicyConfig",
    "PsrlPolicyConfig",
    "RollPolicyConfig",
    "SkyRLPolicyConfig",
    "SlimePolicyConfig",
    "TRLPolicyConfig",
    "VerlPolicyConfig",
    "VerlV1PolicyConfig",
    "preset_config",
]
