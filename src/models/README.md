# Models

This directory contains the implementation of the Temporal Convolutional Network (TCN) model for the Adaptive Hybrid BCI project.

## Overview

The TCN model is the deep learning component (Branch B) of the Adaptive Hybrid BCI architecture. It processes multi-channel epoched EEG data directly to learn complex temporal dynamics.

## Files

- `tcn.py`: Implementation of the TCN model architecture
- `train_tcn_updated.py`: Updated training script that works with preprocessed data
- `validate_tcn.py`: Validation script to test the TCN implementation without training
- `train_tcn.py`: Original training script (placeholder implementation)

## TCN Architecture

The TCN implementation follows the principles of causal, dilated convolutions with residual connections:

1. **Causal Convolutions**: Ensures that predictions are made without information leakage from future time steps
2. **Dilated Convolutions**: Allows the model to have a large receptive field without increasing the number of parameters
3. **Residual Connections**: Helps with gradient flow during training
4. **Temporal Blocks**: Stacks of causal, dilated convolutions with residual connections

### Key Components

- `Chomp1d`: Removes the last elements from the time dimension to make convolutions causal
- `TemporalBlock`: A single temporal block consisting of two causal, dilated convolutions with residual connections
- `TCN`: The main TCN model that stacks multiple temporal blocks

## Usage

### Validating the TCN Implementation

To verify that the TCN model is implemented correctly without training:

```bash
cd src/models
python validate_tcn.py
```

### Training the TCN Model

To train the TCN model on preprocessed data:

```bash
cd src/models
python train_tcn_updated.py --subject_id A01 --epochs 10 --batch_size 32
```

### Options

- `--subject_id`: Subject ID to train on (default: A01)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate for optimizer (default: 0.001)
- `--tcn_output_dim`: Dimension of TCN output feature vector (default: 64)
- `--save_model`: Save the trained model (default: False)

## Model Parameters

For BCI Competition IV Dataset 2a:
- Input channels: 22 (EEG channels)
- Sequence length: 500 (0.5-2.5 seconds at 250 Hz)
- Output dimension: Configurable (default: 64)

Default TCN architecture:
- 4 temporal blocks with 32 filters each
- Kernel size: 3
- Dropout: 0.2

## Requirements

- PyTorch
- NumPy
- scikit-learn