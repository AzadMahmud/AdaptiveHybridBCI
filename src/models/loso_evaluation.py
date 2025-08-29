"""
Leave-One-Subject-Out (LOSO) Evaluation for TCN Model.

This script implements the LOSO cross-validation evaluation for the TCN model,
which is essential for validating subject-independent performance as proposed
in the Adaptive Hybrid BCI approach.

The script will:
1. Train the TCN model on N-1 subjects
2. Test on the left-out subject
3. Repeat for all subjects
4. Report average performance
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import the TCN model and utilities
from src.models.tcn import TCN
import torch.nn as nn
import torch.optim as optim

# Configuration
DATA_DIR = os.path.join(project_root, 'data', 'preprocessed')
RESULTS_DIR = os.path.join(project_root, 'results', 'loso_tcn')
os.makedirs(RESULTS_DIR, exist_ok=True)

# BCI Competition IV Dataset 2a subjects
SUBJECTS = [f"A{str(i).zfill(2)}" for i in range(1, 10)]  # A01 to A09

def load_subject_data(subject_id):
    """
    Load preprocessed data for a specific subject.
    
    Args:
        subject_id (str): Subject identifier (e.g., 'A01').
        
    Returns:
        tuple: (epochs_data, labels) as numpy arrays, or (None, None) if no data.
    """
    epochs_path = os.path.join(DATA_DIR, f"{subject_id}T_epochs.npy")
    labels_path = os.path.join(DATA_DIR, f"{subject_id}T_labels.npy")
    
    if not os.path.exists(epochs_path) or not os.path.exists(labels_path):
        return None, None
    
    epochs_data = np.load(epochs_path)
    labels = np.load(labels_path)
    
    # Handle empty data
    if epochs_data.size == 0 or labels.size == 0:
        return None, None
        
    return epochs_data, labels

def prepare_dataloaders(train_data, train_labels, test_data, test_labels, 
                       batch_size=32, seed=42):
    """
    Prepare train and test DataLoaders.
    
    Args:
        train_data (np.ndarray): Training data.
        train_labels (np.ndarray): Training labels.
        test_data (np.ndarray): Test data.
        test_labels (np.ndarray): Test labels.
        batch_size (int): Batch size.
        seed (int): Random seed.
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Ensure labels are 0-indexed for PyTorch
    train_labels_zero = train_labels - 1
    test_labels_zero = test_labels - 1
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(train_data, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_labels_zero, dtype=torch.long)
    X_test_tensor = torch.tensor(test_data, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_labels_zero, dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, classifier, train_loader, epochs=50, learning_rate=0.001, 
               device=torch.device("cpu")):
    """
    Train the TCN model.
    
    Args:
        model (nn.Module): TCN model.
        classifier (nn.Module): Classifier layer.
        train_loader (DataLoader): Training data loader.
        epochs (int): Number of epochs.
        learning_rate (float): Learning rate.
        device (torch.device): Device to use.
        
    Returns:
        list: Training losses.
    """
    model.train()
    classifier.train()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), 
                          lr=learning_rate)
    
    train_losses = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            features = model(data)
            output = classifier(features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return train_losses

def evaluate_model(model, classifier, test_loader, device=torch.device("cpu")):
    """
    Evaluate the model.
    
    Args:
        model (nn.Module): Trained TCN model.
        classifier (nn.Module): Trained classifier.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device to use.
        
    Returns:
        tuple: (accuracy, kappa)
    """
    model.eval()
    classifier.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            features = model(data)
            output = classifier(features)
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_preds)
    kappa = cohen_kappa_score(all_targets, all_preds)
    
    return accuracy, kappa

def loso_cross_validation(epochs=50, batch_size=32, learning_rate=0.001, 
                         tcn_output_dim=64, num_channels=[32, 32, 32, 32], 
                         kernel_size=3, dropout=0.2):
    """
    Perform Leave-One-Subject-Out cross-validation.
    
    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate.
        tcn_output_dim (int): TCN output dimension.
        num_channels (list): TCN channel configuration.
        kernel_size (int): Kernel size.
        dropout (float): Dropout probability.
        
    Returns:
        dict: Results for each subject.
    """
    print("Starting Leave-One-Subject-Out Cross-Validation for TCN Model")
    print("=" * 60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results = {}
    
    for i, test_subject in enumerate(SUBJECTS):
        print(f"\n[{i+1}/{len(SUBJECTS)}] Testing on subject: {test_subject}")
        
        # Load test subject data
        test_data, test_labels = load_subject_data(test_subject)
        if test_data is None or test_labels is None:
            print(f"  Skipping {test_subject} - no data available")
            results[test_subject] = {'accuracy': 0.0, 'kappa': 0.0, 'status': 'no_data'}
            continue
            
        print(f"  Test data shape: {test_data.shape}")
        
        # Load training data from all other subjects
        train_data_list = []
        train_labels_list = []
        
        train_subjects = [s for s in SUBJECTS if s != test_subject]
        print(f"  Training on subjects: {train_subjects}")
        
        for train_subject in train_subjects:
            data, labels = load_subject_data(train_subject)
            if data is not None and labels is not None:
                train_data_list.append(data)
                train_labels_list.append(labels)
        
        if not train_data_list:
            print(f"  No training data available for {test_subject}")
            results[test_subject] = {'accuracy': 0.0, 'kappa': 0.0, 'status': 'no_train_data'}
            continue
            
        # Concatenate training data
        train_data = np.concatenate(train_data_list, axis=0)
        train_labels = np.concatenate(train_labels_list, axis=0)
        print(f"  Training data shape: {train_data.shape}")
        
        # Prepare data loaders
        train_loader, test_loader = prepare_dataloaders(
            train_data, train_labels, test_data, test_labels, 
            batch_size=batch_size
        )
        
        # Create model
        input_size = 22  # BCI IV 2a channels
        model = TCN(input_size=input_size, output_size=tcn_output_dim,
                   num_channels=num_channels, kernel_size=kernel_size, 
                   dropout=dropout).to(device)
        classifier = nn.Linear(tcn_output_dim, 4).to(device)  # 4 classes
        
        # Train model
        print("  Training model...")
        train_losses = train_model(
            model, classifier, train_loader, 
            epochs=epochs, learning_rate=learning_rate, device=device
        )
        
        # Evaluate model
        print("  Evaluating model...")
        accuracy, kappa = evaluate_model(model, classifier, test_loader, device=device)
        
        print(f"  Results for {test_subject}:")
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    Kappa: {kappa:.4f}")
        
        results[test_subject] = {
            'accuracy': accuracy,
            'kappa': kappa,
            'train_losses': train_losses,
            'status': 'success'
        }
    
    # Calculate and display overall results
    print("\n" + "=" * 60)
    print("LOSO CROSS-VALIDATION RESULTS")
    print("=" * 60)
    
    successful_results = [r for r in results.values() if r['status'] == 'success']
    
    if successful_results:
        accuracies = [r['accuracy'] for r in successful_results]
        kappas = [r['kappa'] for r in successful_results]
        
        print(f"Number of subjects evaluated: {len(successful_results)}")
        print(f"Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Average Kappa: {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")
        
        print("\nPer-subject results:")
        for subject, result in results.items():
            if result['status'] == 'success':
                print(f"  {subject}: Accuracy = {result['accuracy']:.4f}, Kappa = {result['kappa']:.4f}")
            else:
                print(f"  {subject}: {result['status']}")
                
        # Save results
        results_file = os.path.join(RESULTS_DIR, 'loso_tcn_results.txt')
        with open(results_file, 'w') as f:
            f.write("LOSO Cross-Validation Results for TCN Model\n")
            f.write("=" * 50 + "\n")
            f.write(f"Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
            f.write(f"Average Kappa: {np.mean(kappas):.4f} ± {np.std(kappas):.4f}\n\n")
            f.write("Per-subject results:\n")
            for subject, result in results.items():
                if result['status'] == 'success':
                    f.write(f"  {subject}: Accuracy = {result['accuracy']:.4f}, Kappa = {result['kappa']:.4f}\n")
                else:
                    f.write(f"  {subject}: {result['status']}\n")
                    
        print(f"\nResults saved to: {results_file}")
    else:
        print("No successful evaluations completed.")
        
    return results

def main():
    """
    Main function to run LOSO cross-validation.
    """
    # You can adjust these parameters as needed
    results = loso_cross_validation(
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        tcn_output_dim=64,
        num_channels=[32, 32, 32, 32],
        kernel_size=3,
        dropout=0.2
    )
    
    return results

if __name__ == '__main__':
    main()