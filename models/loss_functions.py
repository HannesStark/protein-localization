import torch
import torch.nn.functional as F


def cross_entropy_joint(prediction: torch.Tensor, localization: torch.Tensor,
                        solubility: torch.Tensor, args) -> torch.Tensor:
    localization_loss = F.cross_entropy(prediction[..., :10], localization)
    solubility_loss = F.binary_cross_entropy(prediction[..., -1], solubility)
    return localization_loss + solubility_loss * args.solubility_loss
