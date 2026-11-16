"""HPT prompt model combined with the R-matrix losses (``--method globalHPT``)."""

from hmc.models.global_classifier.hpt.losses import (
    OBJECTIVES,
    hpt_objective,
    mc_logits,
    zmlce,
)
from hmc.models.global_classifier.hpt.model import HPTPromptModel, build_edge_index

__all__ = [
    "OBJECTIVES",
    "HPTPromptModel",
    "build_edge_index",
    "hpt_objective",
    "mc_logits",
    "zmlce",
]
