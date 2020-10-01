import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvsAttendConcat(nn.Module):
    def __init__(self, embeddings_dim: int = 1024, dropout=0.25):
        super(ConvsAttendConcat, self).__init__()

        self.conv1 = nn.Conv1d(embeddings_dim, 512, 21, stride=1, padding=10)
        self.conv2 = nn.Conv1d(embeddings_dim, 512, 15, stride=1, padding=7)
        self.conv3 = nn.Conv1d(embeddings_dim, 512, 9, stride=1, padding=4)
        self.conv4 = nn.Conv1d(embeddings_dim, 512, 5, stride=1, padding=2)
        self.conv5 = nn.Conv1d(embeddings_dim, 512, 3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(embeddings_dim, 512, 1, stride=1, padding=0)

        self.attend = nn.Sequential(nn.Conv1d(embeddings_dim, 512, 1, stride=1), nn.Dropout(dropout))

        self.linear = nn.Sequential(
            nn.Linear(embeddings_dim * 3, 32),
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

        o1 = F.relu(self.conv1(x))  # [batchsize, embeddingsdim/2, seq_len]
        o2 = F.relu(self.conv2(x))
        o3 = F.relu(self.conv3(x))
        o4 = F.relu(self.conv4(x))
        o5 = F.relu(self.conv5(x))
        o6 = F.relu(self.conv6(x))

        attention = F.softmax(self.attend(x), dim=-1)

        o1 = torch.sum(o1 * attention, dim=-1)  # [batchsize, embeddingsdim/2]
        o2 = torch.sum(o2 * attention, dim=-1)
        o3 = torch.sum(o3 * attention, dim=-1)
        o4 = torch.sum(o4 * attention, dim=-1)
        o5 = torch.sum(o5 * attention, dim=-1)
        o6 = torch.sum(o6 * attention, dim=-1)

        o = torch.cat([o1, o2, o3, o4, o5, o6], dim=-1)  # [batchsize, embeddingsdim*3]
        o = self.linear(o)
        return self.output(o)
