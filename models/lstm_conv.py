import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMConv(nn.Module):
    def __init__(self, embeddings_dim: int, output_dim, n_layers, dropout, lstm_hidden_dim: int = 256,
                 kernel_size: int = 9):
        super(LSTMConv, self).__init__()

        self.conv = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1, padding=0)

        self.dropout1 = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embeddings_dim, lstm_hidden_dim, num_layers=n_layers, bidirectional=True, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.output = nn.Linear(32, output_dim)

    def forward(self, x, **kwargs):
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = F.relu(self.conv(x))  # [batchsize, embeddingsdim, seq_len]

        o = self.dropout1(o)
        o = o.permute(2, 0, 1)  # [seq_len, batch_size, embedding_dim]
        # num_directions is 2 for a bidirectional lstm
        # output: [seq_len, batch, num_directions * hidden_size] hidden state of t=seq_len
        # hidden_state: [num_layers * num_directions, batch, hidden_size] hidden state of t=seq_len
        # cell_state: [num_layers * num_directions, batch, hidden_size] cell state of t=seq_len
        output, (hidden_state, cell_state) = self.lstm(o)

        hidden = self.dropout2(
            torch.cat((hidden_state[-2], hidden_state[-1]), dim=1))  # [batch size, lstm_hidden_dim * num directions]

        o = self.linear(hidden)
        return self.output(o)  # [batch_size, output_dim]
