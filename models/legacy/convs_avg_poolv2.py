import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvsAvgPoolv2(nn.Module):
    def __init__(self, embeddings_dim: int = 1024, output_dim: int = 12 , dropout=0.25):
        super(ConvsAvgPoolv2, self).__init__()

        self.conv1 = nn.Conv1d(embeddings_dim, embeddings_dim, 21, stride=1)
        self.conv2 = nn.Conv1d(embeddings_dim, embeddings_dim, 15, stride=1)
        self.conv3 = nn.Conv1d(embeddings_dim, embeddings_dim, 9, stride=1)

        self.linear = nn.Sequential(
            nn.Linear(3 * embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """

        o1 = F.relu(self.conv1(x))  # [batchsize, embeddingsdim, seq_len]
        o2 = F.relu(self.conv2(x))
        o3 = F.relu(self.conv3(x))


        o1 = torch.mean(o1, dim=-1)  # [batchsize, embeddingsdim/2]
        o2 = torch.mean(o2, dim=-1)
        o3 = torch.mean(o3, dim=-1)


        o = torch.cat([o1, o2, o3], dim=-1)  # [batchsize, embeddingsdim*6]

        o = self.linear(o)
        return self.output(o)
