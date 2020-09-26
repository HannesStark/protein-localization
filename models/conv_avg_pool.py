import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAvgPool(nn.Module):
    def __init__(self, n_conv_layers: int = 5):
        super(ConvAvgPool, self).__init__()

        self.conv1 = nn.Conv1d(1024, 256, 3, stride=3)
        self.conv2 = nn.Conv1d(256, 128, 3, stride=2)
        self.conv_layers = nn.ModuleList()
        for i in range(n_conv_layers):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(128, 128, 3, stride=1),
                nn.Sigmoid(),
                nn.BatchNorm1d(128)
            ))

        self.linear = nn.Sequential(
            nn.Linear(256, 32),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.output = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, sequence_length, embeddings_dim] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = F.relu(self.conv1(x))
        #o = F.relu(self.conv2(x))
        #o = torch.sigmoid(self.conv3(o))
        #for conv_layer in self.conv_layers:
        #    o = conv_layer(o)
        o = torch.mean(o, dim=-1)
        #max_pool, _ = torch.max(o, dim=-1)
        #o = torch.cat([avg_pool, max_pool], dim=-1)
        o = self.linear(o)
        return self.output(o)
