"""
TCN Training and Evaluation Script for BCI Competition IV Dataset 2a.

This script demonstrates how to use the TCN model (from tcn.py)
with the preprocessed data obtained from the data_load.ipynb notebook.

It loads data for a single subject, trains the TCN, and evaluates it.
This serves as a baseline for the 'Deep Stream' component of the Hybrid BCI.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

# Add the project root and src/models directory to the Python path
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__)) # /home/azad/Desktop/AdaptiveHybridBCI/src/models
# Get the project root (two levels up)
project_root = os.path.dirname(os.path.dirname(current_dir)) # /home/azad/Desktop/AdaptiveHybridBCI
# Add project root to sys.path
sys.path.insert(0, project_root)

# Import the TCN model directly from the tcn.py file in the same directory
from tcn import TCN # This should work now as tcn.py is in the same dir


# --- Configuration ---
DATA_DIR = os.path.join(project_root, 'data')
SUBJECT_ID = 'A01'
SESSION = 'T' # 'T' for training data

# --- 1. Load Preprocessed Data ---
# This part mimics the output of the data_load.ipynb notebook
# In practice, you would load the epochs_data and labels from a file
# saved by the preprocessing notebook/script.
# For this example, we'll assume it's available in the environment or loaded separately.
# Let's load it directly if it was saved by the notebook (you might need to adjust the path)
# For now, we'll simulate loading it.

def load_preprocessed_data(subject_id, session, data_dir):
    """
    Loads preprocessed epochs and labels.
    This function should be adapted to load the actual output from data_load.ipynb.
    For example, if the notebook saves epochs_data and labels to .npy files:
    """
    # Example: If data_load.ipynb saved files like 'A01T_epochs.npy' and 'A01T_labels.npy'
    # epochs_path = os.path.join(data_dir, f"{subject_id}{session}_epochs.npy")
    # labels_path = os.path.join(data_dir, f"{subject_id}{session}_labels.npy")
    # epochs_data = np.load(epochs_path)
    # labels = np.load(labels_path)
    # return epochs_data, labels
    
    # Placeholder: Simulate data loading
    # You need to replace this with actual data loading logic.
    print("Loading preprocessed data...")
    # This is just a placeholder. Replace with actual data loading.
    # epochs_data shape: (n_trials, n_channels, n_timepoints) e.g., (273, 22, 500)
    # labels shape: (n_trials,) e.g., (273,) with values 1-4
    # For demonstration, we'll generate random data, but you should load real preprocessed data.
    # n_trials, n_channels, n_timepoints = 273, 22, 500
    # epochs_data = np.random.randn(n_trials, n_channels, n_timepoints).astype(np.float32)
    # labels = np.random.randint(1, 5, size=n_trials) # 1-4 for 4 classes
    # print(f"Loaded epochs_data shape: {epochs_data.shape}")
    # print(f"Loaded labels shape: {labels.shape}")
    # return epochs_data, labels
    raise NotImplementedError("Replace this function with actual data loading from preprocessing output.")

# --- 2. Prepare Data for PyTorch ---
def prepare_dataloaders(epochs_data, labels, test_size=0.2, batch_size=32, seed=42):
    """
    Splits data and creates PyTorch DataLoaders.
    """
    print("Preparing data loaders...")
    # Ensure labels are 0-indexed for PyTorch CrossEntropyLoss
    # BCI IV 2a labels are 1-4, PyTorch expects 0-3
    labels_zero_indexed = labels - 1 
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        epochs_data, labels_zero_indexed, test_size=test_size, random_state=seed, stratify=labels_zero_indexed
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long) # Long for classification
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Test loader batches: {len(test_loader)}")
    return train_loader, test_loader, y_test # Return y_test for final evaluation

# --- 3. Train the TCN Model ---
def train_model(model, train_loader, criterion, optimizer, device, epochs=20):
    """
    Trains the TCN model.
    """
    print("Starting training...")
    model.train() # Set model to training mode
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad() # Clear gradients
            output = model(data)  # Forward pass
            loss = criterion(output, target) # Calculate loss
            loss.backward()       # Backward pass
            optimizer.step()      # Update weights
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")

# --- 4. Evaluate the TCN Model ---
def evaluate_model(model, test_loader, device, y_test_original):
    """
    Evaluates the TCN model and prints metrics.
    """
    print("Starting evaluation...")
    model.eval() # Set model to evaluation mode
    all_preds = []
    all_targets = []
    
    with torch.no_grad(): # Disable gradient computation for evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Get the predicted class index
            pred = output.argmax(dim=1, keepdim=True) # shape: (batch_size, 1)
            all_preds.extend(pred.cpu().numpy().flatten()) # Move to CPU and flatten
            all_targets.extend(target.cpu().numpy()) # Move to CPU
            
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    kappa = cohen_kappa_score(all_targets, all_preds)
    
    print(f"Evaluation Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Cohen's Kappa: {kappa:.4f}")
    
    return accuracy, kappa

# --- Main Execution ---
if __name__ == '__main__':
    print("Initializing TCN training and evaluation pipeline...")
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # 1. Load data (replace with actual loading)
        # epochs_data, labels = load_preprocessed_data(SUBJECT_ID, SESSION, DATA_DIR)
        # For demonstration, we'll use dummy data. Replace this section.
        print("WARNING: Using dummy data for demonstration. Replace with actual data loading.")
        n_trials, n_channels, n_timepoints = 200, 22, 500
        epochs_data = np.random.randn(n_trials, n_channels, n_timepoints).astype(np.float32)
        labels = np.random.randint(1, 5, size=n_trials) # 1-4 for 4 classes
        print(f"Dummy epochs_data shape: {epochs_data.shape}")
        print(f"Dummy labels shape: {labels.shape}")
        
        
        # 2. Prepare data loaders
        train_loader, test_loader, y_test_original = prepare_dataloaders(epochs_data, labels, test_size=0.2, batch_size=32)
        
        # 3. Define model, loss, and optimizer
        # TCN parameters
        input_size = 22  # Number of EEG channels
        tcn_output_dim = 64 # Dimension of f_tcn feature vector
        num_channels = [32, 32, 32, 32] # 4 levels with 32 filters
        kernel_size = 3
        dropout = 0.2
        
        model = TCN(input_size=input_size, output_size=tcn_output_dim,
                    num_channels=num_channels, kernel_size=kernel_size, dropout=dropout).to(device)
        
        # Final classifier takes f_tcn as input
        # For 4-class classification
        classifier = nn.Linear(tcn_output_dim, 4).to(device) 
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        # Optimizer for both TCN and classifier
        optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.001)
        
        print(f"TCN Model instantiated.")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params_cls = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        print(f"Trainable parameters in TCN: {total_params}")
        print(f"Trainable parameters in Classifier: {total_params_cls}")
        print(f"Total trainable parameters: {total_params + total_params_cls}")
        
        # 4. Train the model
        train_model(model, train_loader, criterion, optimizer, device, epochs=10) # Use fewer epochs for demo
        
        # 5. Evaluate the model
        # We need to pass the model and the final layer together for evaluation
        # Let's create a combined model for evaluation
        class CombinedModel(nn.Module):
            def __init__(self, tcn, classifier):
                super(CombinedModel, self).__init__()
                self.tcn = tcn
                self.classifier = classifier
            def forward(self, x):
                f_tcn = self.tcn(x)
                out = self.classifier(f_tcn)
                return out
                
        combined_model = CombinedModel(model, classifier)
        evaluate_model(combined_model, test_loader, device, y_test_original)
        
        print("TCN training and evaluation pipeline completed.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()