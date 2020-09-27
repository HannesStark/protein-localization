import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim,output_dim, lstm_hidden_dim, n_layers, dropout):
        super(LSTM, self).__init__()

        # the embedding takes as input the vocab_size and the embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers=n_layers, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(lstm_hidden_dim * 2, output_dim)

    def forward(self, s):
        embedded = self.embedding(s)
        embedded = embedded.permute(1, 0, 2)  # [seq_len, batch_size, embedding_dim]

        # run the LSTM along the sentences of length seq_len
        output, (hidden_state, cell_state) = self.lstm(embedded)

        hidden = self.dropout(
            torch.cat((hidden_state[-2], hidden_state[-1]), dim=1))  # [batch size, lstm_hidden_dim * num directions]

        return self.output(hidden)  # [batch_size, output_dim]
