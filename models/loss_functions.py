from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class JointCrossEntropy(nn.Module):
    def __init__(self, weight=None) -> None:
        super(JointCrossEntropy, self).__init__()
        self.weight = weight

    def forward(self, prediction: Tensor, localization: Tensor,
                solubility: Tensor, solubility_known: bool, args) -> Tuple[Tensor, Tensor, Tensor]:
        """

            Args:
                prediction: output of the network with 12 logits where the last two are for the solubility
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
        localization_loss = F.cross_entropy(prediction[..., :10], localization, weight=self.weight)
        solubility_loss = F.cross_entropy(prediction[..., -2:], solubility, reduction='none')
        solubility_loss = (solubility_loss * solubility_known).mean() * args.solubility_loss
        return localization_loss + solubility_loss, localization_loss, solubility_loss


class LocCrossEntropy(nn.Module):
    def __init__(self, weight=None) -> None:
        super(LocCrossEntropy, self).__init__()
        self.weight = weight

    def forward(self, prediction: Tensor, localization: Tensor,
                solubility: Tensor, solubility_known: bool, args) -> Tuple[Tensor, Tensor, Tensor]:
        """

            Args:
                prediction: output of the network with 12 logits where the last two are for the solubility
                localization: true label for localization


            Returns:
                loss: the overall loss

            """
        localization_loss = F.cross_entropy(prediction[..., :10], localization, weight=self.weight)
        return localization_loss, localization_loss, torch.tensor([0])
