from typing import Tuple

import torch
import torch.nn.functional as F


def cross_entropy_joint(prediction: torch.Tensor, localization: torch.Tensor,
                        solubility: torch.Tensor, solubility_known: bool, args) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """

    Args:
        prediction: output of the network with 11 logits where the last one is for the solubility
        localization: true label for localization
        solubility: true label for
        solubility_known: tensor on device whether or not the solubility is known such that the solubility loss is set
        to 0 the solubility is unknown.
        args: training arguments containing the weighting for the solubility loss

    Returns:
        loss: the overall loss
        loc_loss: loss of localization information
        sol_loss: loss of the solubility prediction

    """
    localization_loss = F.cross_entropy(prediction[..., :10], localization)
    solubility_loss = F.binary_cross_entropy_with_logits(prediction[..., -1], solubility, reduction='none')
    solubility_loss = (solubility_loss * solubility_known).sum() * args.solubility_loss
    return localization_loss + solubility_loss, localization_loss, solubility_loss
