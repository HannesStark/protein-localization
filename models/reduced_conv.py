import torch
import torch.nn as nn


class ReducedConv(nn.Module):
    def __init__(self, embeddings_dim: int = 1024, hidden_dim: int = 32, output_dim: int = 11, dropout: float = 0.25):
        """
        Simple Feed forward model with default parameters like the network tha is ued in the SeqVec paper.
        Args:
            embeddings_dim: dimension of the input
            hidden_dim: dimension of the hidden layers
            output_dim: output dimension (number of classes that should be classified)
            number_hidden_layers: number of hidden layers (0 by default)
            dropout: dropout ratio of every layer
        """
        super(ReducedConv, self).__init__()

        self.conv1 = nn.Conv1d(embeddings_dim, embeddings_dim, 3, stride=1)
        self.conv2 = nn.Conv1d(embeddings_dim, embeddings_dim, 3, stride=1)
        self.conv2 = nn.Conv1d(embeddings_dim, embeddings_dim, 3, stride=2)

        self.linear = nn.Sequential(
            nn.Linear(embeddings_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.output = nn.Linear(32, output_dim)

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        x = x[:,None,:]
        print(x.shape)
        o = x
        o = self.conv1(o)
        o = self.conv2(o)
        o = torch.cat([o, x], dim=-1)
        o = self.input(o)
        for hidden_layer in self.hidden:
            o = hidden_layer(o)
        return self.output(o)
