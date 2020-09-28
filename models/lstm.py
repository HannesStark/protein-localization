import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, embedding_dim, output_dim, lstm_hidden_dim, n_layers, dropout):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers=n_layers, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(lstm_hidden_dim * 2, output_dim)

    def forward(self, x):
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        x = x.permute(2, 0, 1)  # [seq_len, batch_size, embedding_dim]

        # num_directions is 2 for a bidirectional lstm
        # hidden_state: [num_layers * num_directions, batch, hidden_size] hidden state of t=seq_len
        # cell_state: [num_layers * num_directions, batch, hidden_size] cell state of t=seq_len
        output, (hidden_state, cell_state) = self.lstm(x)

        hidden = self.dropout(
            torch.cat((hidden_state[-2], hidden_state[-1]), dim=1))  # [batch size, lstm_hidden_dim * num directions]

        return self.output(hidden)  # [batch_size, output_dim]
