import torch
import torch.nn as nn


class BiLSTMEmotionClassifier(nn.Module):

    def __init__(self, input_size=40,
                 hidden_size=128,
                 num_layers=2,
                 num_classes=8):

        super(BiLSTMEmotionClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):

        lstm_out, _ = self.lstm(x)

        output = lstm_out[:, -1, :]

        output = self.fc(output)

        return output