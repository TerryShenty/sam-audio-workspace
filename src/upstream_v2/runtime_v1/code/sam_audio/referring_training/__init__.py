"""Experimental referring-prompt fine-tuning for SAM Audio.

The public SAM Audio release contains inference code but not the original training
pipeline.  This package keeps the released model path intact and adds the minimum
components needed for a reproducible referring-prompt experiment.
"""

from .losses import (
    cosine_alignment_loss,
    flow_matching_loss,
    referring_contrastive_loss,
    sample_flow_matching_state,
)

__all__ = [
    "cosine_alignment_loss",
    "flow_matching_loss",
    "referring_contrastive_loss",
    "sample_flow_matching_state",
]
