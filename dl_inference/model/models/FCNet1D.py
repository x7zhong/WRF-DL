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


class block(nn.Module):
    def __init__(self, input_size, output_size):
        super(block, self).__init__()
        # Our first linear layer take input_size, in this case 784 nodes to 50
        # and our second linear layer takes 50 to the num_classes we have, in
        # this case 10.

        self.fc1 = nn.Linear(input_size, output_size)
        self.bn1 = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x here is the mnist images and we run it through fc1, fc2 that we created above.
        we also add a ReLU activation function in between and for that (since it has no parameters)
        I recommend using nn.functional (F)
        """

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class FCNet1D(nn.Module):
    def __init__(self, feature_channel, output_channel,
                 hidden_number, hidden_size,
                 signal_length=55, dim_add=0):
        super(FCNet1D, self).__init__()
        self.feature_channel = feature_channel
        self.output_channel = output_channel
        self.signal_length = signal_length

        self.first_layer = block(feature_channel * signal_length, hidden_size)

        layers = []
        for i in range(hidden_number):
            layers.append(block(hidden_size, hidden_size))
        self.mediate_layer = nn.Sequential(*layers)

        self.last_layer = nn.Linear(
            hidden_size, signal_length * output_channel)

        self.dim_add = dim_add
        self.dim_change = None
        if(self.dim_add > 0):
            self.dim_change = DimChange(
                signal_length, signal_length + self.dim_add)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.first_layer(x)
        x = self.mediate_layer(x)
        x = self.last_layer(x)
        x = x.reshape([x.size(0), self.output_channel, -1])
        if(self.dim_change):
            x = self.dim_change(x)
        return x


def test():
    # net = FCNet1D(feature_channel=20,
    #               output_channel=4,
    #               hidden_number=5,
    #               hidden_size=100,
    #               signal_length=56,
    #               dim_add=1
    #               )

    net = FCNet1D(feature_channel=34,
                output_channel=4,
                # hidden_number=10,
                # hidden_size=200,
                hidden_number=3,
                hidden_size=50,
                signal_length=57,
                dim_add=0
                )

    print(net)
    y = net(torch.randn(1000, 34, 57))
    print(y.size())
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(pytorch_total_params)


if __name__ == "__main__":
    test()
