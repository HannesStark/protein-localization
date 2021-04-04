import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvsAvgPool(nn.Module):
    def __init__(self, embeddings_dim: int = 1024, output_dim: int = 12 , dropout=0.25):
        super(ConvsAvgPool, self).__init__()

        self.conv1 = nn.Conv1d(embeddings_dim, embeddings_dim, 21, stride=1)
        self.conv2 = nn.Conv1d(embeddings_dim, embeddings_dim, 15, stride=1)
        self.conv3 = nn.Conv1d(embeddings_dim, embeddings_dim, 9, stride=1)


        self.linear = nn.Sequential(
            nn.Linear(embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor, mask, sequence_lengths, frequencies) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """

        o1 = F.relu(self.conv1(x))  # [batchsize, embeddingsdim, seq_len]
        o2 = F.relu(self.conv2(x))
        o3 = F.relu(self.conv3(x))


        o = torch.cat([o1, o2, o3], dim=-1)  # [batchsize, embeddingsdim, seq_len*6]
        o = torch.mean(o, dim=-1)

        o = self.linear(o)
        return self.output(o)
