#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:54:07 2021

@author: yaoyichen
"""
import torch
import torch.nn as nn


class DimChange(nn.Module):
    """
    B*V*H  -> B*V*(H + H_add)
    """

    def __init__(self, channel_number, output_number):
        super(DimChange, self).__init__()
        self.channel_number = channel_number
        self.output_number = output_number
        self.conv_final = nn.Conv1d(
            channel_number, output_number, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.conv_final(x)
        x = torch.permute(x, (0, 2, 1))
        return x
