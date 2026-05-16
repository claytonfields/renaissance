"""
Task registry (Phase 3).

`TASK_REGISTRY` maps a task name → singleton `Task`. Adding a new
objective is now: write one `Task` subclass, add one line here. No edits
to a model `__init__` or `forward` (those become registry loops in
Phase 4).

Only objectives with a real legacy `compute_*` are ported: mlm, itm,
vqa, nlvr2, snli, ref, ref2, mrpc. `irtr` (needs `draw_false_text`
negatives + a rank head derived from ITM weights) and the stub GLUE
tasks (rte/wnli/sst2/qqp/qnli/mnli/cola — heads but no objective) are
intentionally out of scope; the rewrite drops the dead branches.
"""

from typing import Dict

from .base import Task, TaskOutput
from .itm import ItmTask
from .mlm import MlmTask
from .mrpc import MrpcTask
from .nlvr2 import Nlvr2Task
from .ref import RefTask
from .ref2 import Ref2Task
from .snli import SnliTask
from .vqa import VqaTask

TASK_REGISTRY: Dict[str, Task] = {
    t.name: t
    for t in (
        MlmTask(),
        ItmTask(),
        VqaTask(),
        Nlvr2Task(),
        SnliTask(),
        RefTask(),
        Ref2Task(),
        MrpcTask(),
    )
}


def get_task(name: str) -> Task:
    try:
        return TASK_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown task {name!r}. Registered: {sorted(TASK_REGISTRY)}"
        ) from None


__all__ = ["Task", "TaskOutput", "TASK_REGISTRY", "get_task"]
