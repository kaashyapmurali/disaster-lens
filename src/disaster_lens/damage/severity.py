"""Ordinal summaries of four-class damage probabilities, written alongside exported predictions."""

import torch
from torch import Tensor

from disaster_lens.data.manifest import DAMAGE_LABELS

SEVERITY_COLUMNS = ("severity_index", "probability_at_least_minor", "probability_at_least_major", "probability_destroyed")


def severity_outputs(probabilities: Tensor) -> Tensor:
    """Return [N, 4]: expected rank / 3, P(Y >= minor), P(Y >= major), P(Y = destroyed).

    Summing tails avoids subtraction cancellation near probability one. The index is a conventional
    equal-boundary ordinal summary (0 = no damage, 1 = destroyed), not percent damage.
    """
    if probabilities.ndim != 2 or probabilities.shape[1] != len(DAMAGE_LABELS):
        raise ValueError("Expected probabilities with shape [N, 4].")
    exceedance = probabilities[:, 1:].flip(dims=(1,)).cumsum(dim=1).flip(dims=(1,))
    return torch.cat((exceedance.mean(dim=1, keepdim=True), exceedance), dim=1)
