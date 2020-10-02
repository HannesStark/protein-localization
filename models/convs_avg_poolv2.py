import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvsAvgPoolv2(nn.Module):
    def __init__(self, embeddings_dim: int = 1024, dropout=0.25):
        super(ConvsAvgPoolv2, self).__init__()

        self.conv1 = nn.Conv1d(embeddings_dim, embeddings_dim, 21, stride=1)
        self.conv2 = nn.Conv1d(embeddings_dim, embeddings_dim, 15, stride=1)
        self.conv3 = nn.Conv1d(embeddings_dim, embeddings_dim, 9, stride=1)
        self.conv4 = nn.Conv1d(embeddings_dim, embeddings_dim, 5, stride=1)
        self.conv5 = nn.Conv1d(embeddings_dim, embeddings_dim, 3, stride=1)
        self.conv6 = nn.Conv1d(embeddings_dim, embeddings_dim, 1, stride=1)

        self.linear = nn.Sequential(
            nn.Linear(6 * embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.output = nn.Linear(32, 11)

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
        o4 = F.relu(self.conv4(x))
        o5 = F.relu(self.conv5(x))
        o6 = F.relu(self.conv6(x))

        o1 = torch.mean(o1, dim=-1)  # [batchsize, embeddingsdim/2]
        o2 = torch.mean(o2, dim=-1)
        o3 = torch.mean(o3, dim=-1)
        o4 = torch.mean(o4, dim=-1)
        o5 = torch.mean(o5, dim=-1)
        o6 = torch.mean(o6, dim=-1)

        o = torch.cat([o1, o2, o3, o4, o5, o6], dim=-1)  # [batchsize, embeddingsdim*6]

        o = self.linear(o)
        return self.output(o)
