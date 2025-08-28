"""
Temporal Convolutional Network (TCN) for Motor Imagery EEG Classification.

This module defines a TCN model suitable for processing epoched EEG data
as produced by the preprocessing pipeline (e.g., data_load.ipynb).
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
    # """
    # A single Temporal Block of the TCN.
    # Consists of two causal, dilated convolutions with residual connections.
    # """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        Args:
            n_inputs (int): Number of input channels.
            n_outputs (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride for the convolution.
            dilation (int): Dilation factor for the convolution.
            padding (int): Padding applied to make convolution causal.
            dropout (float): Dropout probability.
        """
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

        # Residual connection
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Down-projection for residual if dimensions don't match
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights for conv layers."""
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)

    def forward(self, x):
        """Forward pass through the temporal block."""
        out = self.net(x)
        # Apply residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    The main TCN model.
    Stacks multiple Temporal Blocks.
    Outputs a feature vector for each trial.
    """
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            input_size (int): Number of input channels (e.g., 22 for EEG).
            output_size (int): Dimension of the final output feature vector (f_tcn).
            num_channels (list): List of hidden channel sizes for each level.
                                E.g., [25, 25, 25, 25] for 4 levels with 25 channels.
            kernel_size (int): Kernel size for convolutions.
            dropout (float): Dropout probability.
        """
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            # Calculate dilation for this level
            dilation_size = 2 ** i
            # Input channels for this layer
            in_channels = input_size if i == 0 else num_channels[i-1]
            # Output channels for this layer
            out_channels = num_channels[i]
            # Padding to make convolution causal
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        
        # Final layer to produce the output feature vector f_tcn
        # We use AdaptiveAvgPool1d to handle variable sequence lengths after convolutions
        # and then a linear layer to map to the desired output size.
        self.pool = nn.AdaptiveAvgPool1d(1) # Global average pooling
        self.flatten = nn.Flatten()
        # The input to the linear layer will be the number of channels from the last TCN layer
        last_layer_channels = num_channels[-1] if num_channels else input_size
        self.fc_out = nn.Linear(last_layer_channels, output_size)
        
        # Initialize final layer weights
        nn.init.kaiming_normal_(self.fc_out.weight)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (Tensor): Input tensor of shape (batch_size, n_channels, seq_length).
                        For BCI IV 2a epoched data: (batch_size, 22, 500).
        Returns:
            Tensor: Output feature vector of shape (batch_size, output_size).
        """
        # x shape: (batch_size, n_channels, seq_length)
        out = self.network(x)
        # out shape: (batch_size, num_channels[-1], new_seq_length)
        out = self.pool(out)
        # out shape: (batch_size, num_channels[-1], 1)
        out = self.flatten(out)
        # out shape: (batch_size, num_channels[-1])
        out = self.fc_out(out)
        # out shape: (batch_size, output_size)
        return out

# --- Example Usage ---
# This would typically be in a separate training script or notebook cell.

# if __name__ == '__main__':
#     # Example parameters for BCI IV 2a data
#     batch_size = 32
#     n_channels = 22 # Number of EEG channels
#     seq_length = 500 # Number of time points per epoch (0.5-2.5s at 250Hz)
#     tcn_output_dim = 64 # Dimension of the f_tcn feature vector
    
#     # Define TCN architecture
#     # E.g., 4 levels with 32 filters each, kernel size 3
#     num_channels = [32, 32, 32, 32] 
#     kernel_size = 3
    
#     # Create the model
#     model = TCN(input_size=n_channels, output_size=tcn_output_dim,
#                 num_channels=num_channels, kernel_size=kernel_size, dropout=0.2)
    
#     # Create dummy input data (like one batch from DataLoader)
#     # Shape: (batch_size, n_channels, seq_length)
#     x = torch.randn(batch_size, n_channels, seq_length)
    
#     # Forward pass
#     f_tcn = model(x)
    
#     print(f"Input shape: {x.shape}")
#     print(f"Output feature vector (f_tcn) shape: {f_tcn.shape}") # Should be (batch_size, tcn_output_dim)
    
#     # Number of parameters
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total trainable parameters: {total_params}")