import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionNet(nn.Module):
    def __init__(self):
        super(AttentionNet, self).__init__()

        self.conv1 = nn.Conv1d(1024, 1024, 1, stride=1)

        self.linear = nn.Sequential(
            nn.Linear(1024, 32),
            nn.Dropout(0.3),
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
        o = self.conv1(x)
        attention = F.softmax(o, dim=-1)
        o = torch.sum(x * attention, dim=-1)
        o = self.linear(o)
        return self.output(o)
