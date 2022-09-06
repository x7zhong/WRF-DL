#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:54:07 2021

@author: yaoyichen
"""
import torch
import torch.nn as nn
import sys
sys.path.append("..")
from models.DimChangeModule import DimChange


class RNN_LSTM(nn.Module):
    def __init__(self, feature_channel, output_channel, hidden_size,
                 num_layers):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_channel = output_channel
        self.lstm = nn.LSTM(feature_channel, hidden_size,
                            num_layers, batch_first=True, bidirectional=True)

        self.final = nn.Conv1d(
            2 * hidden_size, output_channel, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = torch.permute(x, [0, 2, 1])
        # Set initial hidden and cell states

        h0 = torch.zeros(2*self.num_layers, x.shape[0],
                         self.hidden_size, requires_grad=False).to(x.device)
        c0 = torch.zeros(2*self.num_layers, x.shape[0],
                         self.hidden_size, requires_grad=False).to(x.device)

        hidden = (h0, c0)
        (h0, c0) = hidden

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )
        out = torch.permute(out, [0, 2, 1])
        out = self.final(out)

        return out


class RNN_GRU(nn.Module):
    def __init__(self, feature_channel, output_channel, hidden_size,
                 num_layers):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_channel = output_channel
        self.lstm = nn.GRU(feature_channel, hidden_size,
                           num_layers, batch_first=True, bidirectional=True)

        self.final = nn.Conv1d(
            2 * hidden_size, output_channel, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = torch.permute(x, [0, 2, 1])
        # Set initial hidden and cell states
        h0 = torch.zeros(2*self.num_layers, x.shape[0],
                         self.hidden_size, requires_grad=False).to(x.device)

        out, _ = self.lstm(x, h0)
        out = torch.permute(out, [0, 2, 1])
        out = self.final(out)
        return out


def test():
    hidden_size = 16
    num_layers = 1
    batch_size = 100

    net = RNN_LSTM(feature_channel=34, output_channel=4, hidden_size=hidden_size,
                   num_layers=num_layers)

    # net = RNN_GRU(feature_channel=20, output_channel=4, hidden_size=hidden_size,
    #               num_layers=num_layers)

    y = net(torch.randn(batch_size, 34, 54))
    print(y.size())

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(pytorch_total_params)


if __name__ == "__main__":
    test()
