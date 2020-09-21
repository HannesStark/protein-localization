import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 32, output_dim: int = 10, number_hidden_layers: int = 0,
                 dropout: float = 0.25):
        """
        Simple Feed forward model with default parameters like the network tha is ued in the SeqVec paper.
        Args:
            input_dim: dimension of the input
            hidden_dim: dimension of the hidden layers
            output_dim: output dimension (number of classes that should be classified)
            number_hidden_layers: number of hidden layers (0 by default)
            dropout: dropout ratio of every layer
        """
        super(FFN, self).__init__()

        self.number_hidden_layers = number_hidden_layers
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.hidden = nn.ModuleList()
        for i in range(self.number_hidden_layers):
            self.hidden.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ))
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.input(x)
        for hidden_layer in self.hidden:
            o = hidden_layer(o)
        return self.output(o)
