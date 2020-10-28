import torch
import torch.nn as nn


class FFNLateConcat(nn.Module):
    def __init__(self, embeddings_dim: int = 1024, output_dim: int = 12, hidden_dim: int = 32, dropout: float = 0.25):
        """
        Simple Feed forward model with default parameters like the network tha is ued in the SeqVec paper.
        Args:
            embeddings_dim: dimension of the input
            hidden_dim: dimension of the hidden layers
            output_dim: output dimension (number of classes that should be classified)
            dropout: dropout ratio of every layer
        """
        super(FFNLateConcat, self).__init__()

        assert embeddings_dim % 2 == 0, 'Are you using concatenated embeddings?'

        self.linear1 = nn.Sequential(
            nn.Linear(embeddings_dim // 2, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(embeddings_dim // 2, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )

        self.output = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, mask) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        embeddings_dim = x.shape[-1]
        o1 = self.linear1(x[:, :embeddings_dim // 2])
        o2 = self.linear2(x[:, embeddings_dim // 2:])
        o = torch.cat([o1, o2], dim=-1)
        return self.output(o)
