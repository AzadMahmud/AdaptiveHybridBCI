"""
Temporal Convolutional Network (TCN) for Motor Imagery EEG Classification.

This module defines a TCN model suitable for processing epoched EEG data
as produced by the preprocessing pipeline.
It follows the principles of causal, dilated convolutions with residual connections.

Adapted for BCI Competition IV 2a (22 EEG channels, 500 time points).
"""

import torch
import torch.nn as nn
import numpy as np


class Chomp1d(nn.Module):
    """
    Removes the last `chomp_size` elements from the last dimension (time).
    This is used to make convolutions causal.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ATCNet(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(ATCNet, self).__init__()
        self.conv_block = nn.Sequential(
            ConvBlock(1, 16, (1, 64), (1, 1), (0, 32)),
            nn.MaxPool2d((1, 4))
        )

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = 16 * 22 if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.tcn_network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        last_layer_channels = num_channels[-1] if num_channels else 16 * 22
        self.fc_out = nn.Linear(last_layer_channels, output_size)
        nn.init.kaiming_normal_(self.fc_out.weight)

    def forward(self, x):
        x = x.unsqueeze(1) # Add a dimension for the convolutional block
        x = self.conv_block(x)
        x = x.view(x.size(0), -1, x.size(3))
        x = self.tcn_network(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc_out(x)
        return x