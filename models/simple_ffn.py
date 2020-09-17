from typing import Tuple

import torch
import torch.nn as nn


class SimpleFFN(nn.Module):
    def __init__(self, binary_output: bool = False, use_batch_norm=True):
        """ Simple Feed forward model as used in the SeqVec paper but with only one output logit for membrane classification
        instead of two logits.

        Args:
            binary_output: whether the network is used for localization (output of shape [batch_size, 10]) or for membrane
            classification (output of shape [batch_size, 1]). Default is localization.
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

        if binary_output:
            self.classification_logits = nn.Linear(32, 1)
        else:
            self.classification_logits = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: [batch_size, 1024] embedding tensor that should be classified into ten or two classes

        Returns:
            classification: [batch_size,10] or [batch_size,1] depending on whether or not binary_classification is
            set to true

        """
        out = self.layer(x)

        return self.classification_logits(out)
