import pandas as pd
import logging
from torch import nn
import torch
import numpy as np


class RecurrentAndNeural(nn.Module):

    def __init__(self, input_size, hidden_size, num_factors, bidirectional):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = 0.2
        self.bidirectional = bidirectional
        # 根据bidirectional参数调整GRU层
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=bidirectional)
        # 输出层
        rnn_output_size = hidden_size * (2 if bidirectional else 1)
        self.out = nn.Sequential(
            nn.BatchNorm1d(rnn_output_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(rnn_output_size, num_factors),
            nn.BatchNorm1d(num_factors)
        )

    def forward(self, x):
        gru_out = self.gru(x)[0]
        gru_out = zscore_norm(gru_out, dim=0)
        out1 = self.out(gru_out[:, -1, :])

        return out1


def zscore_norm(tensor, dim=0):
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True)
    return (tensor - mean) / (std + 1e-5)
