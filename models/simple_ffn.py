from typing import Tuple

import torch
import torch.nn as nn


class SimpleFFN(nn.Module):
    def __init__(self, use_batch_norm=True):
        """ Simple Feed forward model as used in the SeqVec paper but with only one output logit for membrane classification
        instead of two logits.

        Args:
            use_batch_norm: whether or not to include Batchnorm after nonlinearity
        """
        super(SimpleFFN, self).__init__()
        if use_batch_norm:
            self.layer = nn.Sequential(
                nn.Linear(1024, 32),
                nn.Dropout(0.25),
                nn.ReLU(),
                nn.BatchNorm1d(32)
            )
        else:
            self.layer = nn.Sequential(
                nn.Linear(1024, 32),
                nn.Dropout(0.25),
                nn.ReLU(),
            )
        self.localization_classifier = nn.Linear(32, 10)
        self.membrane_classifier = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x: [1024] embedding tensor that should be classified into ten or two classes

        Returns:
            localization: [10] encoding of localization class
            membrane: [1] encoding of wether the protein is in the membrane or not

        """
        out = self.layer(x)
        localization = self.localization_classifier(out)
        membrane = self.membrane_classifier(out)

        return localization, membrane
