import torch
import torch.nn.functional as F


def cross_entropy_joint(prediction: torch.Tensor, localization: torch.Tensor,
                        solubility: torch.Tensor, solubility_known: bool, args) -> torch.Tensor:
    localization_loss = F.cross_entropy(prediction[..., :10], localization)
    solubility_loss = F.binary_cross_entropy_with_logits(prediction[..., -1], solubility, reduction='none')
    solubility_loss = (solubility_loss * solubility_known).sum()
    return localization_loss + solubility_loss * args.solubility_loss
