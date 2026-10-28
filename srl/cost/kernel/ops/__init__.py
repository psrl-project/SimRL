"""One class per kernel op, each owning its schema, SOL formula, and floor policy."""

# Importing every op module here is what populates `base.OP_REGISTRY`. `op_for` is called from
# modules with no reason to know which ops exist: the perf table writer, `database.py`,
# `floor.py`, so registration cannot be left to each caller remembering to import the right
# module first, which would make the registry's contents depend on import order. Ruff cannot
# see an import-time side effect, hence the noqa.
from . import attention as _attention_ops  # noqa: F401
from . import attention_training as _attention_training_ops  # noqa: F401
from . import collective as _collective_ops  # noqa: F401
from . import elementwise as _elementwise_ops  # noqa: F401
from . import gdn as _gdn_ops  # noqa: F401
from . import gemm as _gemm_ops  # noqa: F401
from . import moe as _moe_ops  # noqa: F401
from . import optimizer as _optimizer_ops  # noqa: F401
