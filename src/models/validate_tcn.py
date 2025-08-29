"""
Validation script for the TCN model.

This script tests the TCN model implementation without training, 
using dummy data to verify the architecture works correctly. 
It's useful for checking the model implementation without requiring a GPU.

Usage:
    cd src/models
    python validate_tcn.py
"""

import sys
import os
import torch
import numpy as np

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import the TCN model
from src.models.tcn import TCN

def test_tcn_architecture():
    """
    Test the TCN architecture with dummy data to verify it works correctly.
    """
    print("Testing TCN architecture...")
    
    # Device (use CPU for validation)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Example parameters for BCI IV 2a data
    batch_size = 32
    n_channels = 22  # Number of EEG channels
    seq_length = 500  # Number of time points per epoch (0.5-2.5s at 250Hz)
    tcn_output_dim = 64  # Dimension of the f_tcn feature vector
    
    # Define TCN architecture
    # E.g., 4 levels with 32 filters each, kernel size 3
    num_channels = [32, 32, 32, 32] 
    kernel_size = 3
    
    # Create the model
    print("Creating TCN model...")
    model = TCN(input_size=n_channels, output_size=tcn_output_dim,
                num_channels=num_channels, kernel_size=kernel_size, dropout=0.2).to(device)
    
    # Create dummy input data (like one batch from DataLoader)
    # Shape: (batch_size, n_channels, seq_length)
    print("Creating dummy input data...")
    x = torch.randn(batch_size, n_channels, seq_length).to(device)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    print("Performing forward pass...")
    f_tcn = model(x)
    
    print(f"Output feature vector (f_tcn) shape: {f_tcn.shape}")  # Should be (batch_size, tcn_output_dim)
    
    # Number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    
    # Verify output shape
    expected_shape = (batch_size, tcn_output_dim)
    if f_tcn.shape == expected_shape:
        print("SUCCESS: TCN model output shape is correct!")
        return True
    else:
        print(f"ERROR: TCN model output shape is incorrect. Expected {expected_shape}, got {f_tcn.shape}")
        return False

def test_tcn_with_small_data():
    """
    Test the TCN with a smaller batch size to verify it works with different input sizes.
    """
    print("\nTesting TCN with smaller batch size...")
    
    # Device (use CPU for validation)
    device = torch.device("cpu")
    
    # Smaller parameters
    batch_size = 8
    n_channels = 22  # Number of EEG channels
    seq_length = 500  # Number of time points per epoch
    tcn_output_dim = 32  # Smaller output dimension
    
    # Define TCN architecture
    num_channels = [16, 16, 16]  # Fewer layers for faster testing
    kernel_size = 2
    
    # Create the model
    model = TCN(input_size=n_channels, output_size=tcn_output_dim,
                num_channels=num_channels, kernel_size=kernel_size, dropout=0.1).to(device)
    
    # Create dummy input data
    x = torch.randn(batch_size, n_channels, seq_length).to(device)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    f_tcn = model(x)
    
    print(f"Output feature vector (f_tcn) shape: {f_tcn.shape}")
    
    # Verify output shape
    expected_shape = (batch_size, tcn_output_dim)
    if f_tcn.shape == expected_shape:
        print("SUCCESS: TCN model works with smaller batch size!")
        return True
    else:
        print(f"ERROR: TCN model output shape is incorrect. Expected {expected_shape}, got {f_tcn.shape}")
        return False

def test_tcn_gradient_flow():
    """
    Test that gradients flow properly through the TCN model.
    """
    print("\nTesting TCN gradient flow...")
    
    # Device (use CPU for validation)
    device = torch.device("cpu")
    
    # Parameters
    batch_size = 4
    n_channels = 22
    seq_length = 500
    tcn_output_dim = 32
    
    # Define TCN architecture
    num_channels = [16, 16]  # Even fewer layers for faster testing
    kernel_size = 2
    
    # Create the model
    model = TCN(input_size=n_channels, output_size=tcn_output_dim,
                num_channels=num_channels, kernel_size=kernel_size, dropout=0.0).to(device)
    
    # Create dummy input data and labels
    x = torch.randn(batch_size, n_channels, seq_length, requires_grad=True).to(device)
    labels = torch.randint(0, 4, (batch_size,)).to(device)  # 4 classes
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Final classifier
    classifier = torch.nn.Linear(tcn_output_dim, 4).to(device)
    
    # Forward pass
    f_tcn = model(x)
    outputs = classifier(f_tcn)
    
    # Compute loss
    loss = criterion(outputs, labels)
    
    # Backward pass
    loss.backward()
    
    # Check if gradients exist
    has_gradients = True
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"WARNING: No gradient for parameter {name}")
            has_gradients = False
        elif param.grad.sum() == 0:
            print(f"WARNING: Zero gradient for parameter {name}")
            has_gradients = False
            
    if has_gradients:
        print("SUCCESS: Gradients flow properly through the TCN model!")
        return True
    else:
        print("ERROR: Issues with gradient flow in the TCN model.")
        return False

def main():
    """
    Main function to run all validation tests.
    """
    print("Validating TCN model implementation...")
    print("=" * 50)
    
    # Run all tests
    test1_passed = test_tcn_architecture()
    test2_passed = test_tcn_with_small_data()
    test3_passed = test_tcn_gradient_flow()
    
    print("\n" + "=" * 50)
    print("Validation Results:")
    print(f"TCN Architecture Test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Small Data Test: {'PASSED' if test2_passed else 'FAILED'}")
    print(f"Gradient Flow Test: {'PASSED' if test3_passed else 'FAILED'}")
    
    all_passed = test1_passed and test2_passed and test3_passed
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    if all_passed:
        print("\nThe TCN model implementation is working correctly!")
        print("You can now proceed with training using the train_tcn_updated.py script.")
    else:
        print("\nThere are issues with the TCN model implementation.")
        print("Please check the error messages above.")

if __name__ == '__main__':
    main()