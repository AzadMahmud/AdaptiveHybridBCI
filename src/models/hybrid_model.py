"""
Adaptive Hybrid BCI Model.

This module defines the adaptive hybrid BCI model, which combines a handcrafted
feature stream with a deep learning stream (ATCNet) using an attention-based
fusion mechanism.
"""

import torch
import torch.nn as nn
from .tcn import ATCNet

class AdaptiveHybridBCI(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout, num_handcrafted_features):
        super(AdaptiveHybridBCI, self).__init__()
        
        # Handcrafted feature stream
        self.handcrafted_fc = nn.Linear(num_handcrafted_features, 64)
        
        # Deep learning stream
        self.atcnet = ATCNet(input_size, 64, num_channels, kernel_size, dropout)
        
        # Attention mechanism
        self.attention_fc = nn.Linear(128, 2)
        
        # Final classifier
        self.classifier = nn.Linear(64, 4)

    def forward(self, x_eeg, x_handcrafted):
        # Handcrafted features
        f_expert = self.handcrafted_fc(x_handcrafted)
        
        # Deep learning features
        f_tcn = self.atcnet(x_eeg)
        
        # Attention mechanism
        attention_input = torch.cat((f_expert, f_tcn), dim=1)
        attention_weights = torch.softmax(self.attention_fc(attention_input), dim=1)
        
        # Weighted fusion
        f_fused = attention_weights[:, 0].unsqueeze(1) * f_expert + attention_weights[:, 1].unsqueeze(1) * f_tcn
        
        # Classification
        output = self.classifier(f_fused)
        
        return output
