"""
Verification script for the preprocessed data.

This script checks the shape and content of the preprocessed data files
to ensure they were generated correctly.
"""

import os
import numpy as np

def verify_preprocessed_data(data_dir="data/preprocessed"):
    """
    Verify the preprocessed data files.
    
    Args:
        data_dir (str): Path to the directory containing preprocessed data.
    """
    # Resolve relative path to absolute path
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', data_dir))
        
    print("Verifying preprocessed data...")
    print(f"Data directory: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return False
    
    # BCI Competition IV Dataset 2a subjects
    subjects = [f"A{str(i).zfill(2)}" for i in range(1, 10)]  # A01 to A09
    sessions = ['T', 'E']  # Training and Evaluation
    
    total_files = 0
    verified_files = 0
    
    for subject in subjects:
        for session in sessions:
            epochs_file = os.path.join(data_dir, f"{subject}{session}_epochs.npy")
            labels_file = os.path.join(data_dir, f"{subject}{session}_labels.npy")
            
            total_files += 2
            
            # Check if files exist
            if not os.path.exists(epochs_file):
                print(f"Missing epochs file: {epochs_file}")
                continue
                
            if not os.path.exists(labels_file):
                print(f"Missing labels file: {labels_file}")
                continue
            
            # Load and verify data
            try:
                epochs = np.load(epochs_file)
                labels = np.load(labels_file)
                
                # Check shapes
                if epochs.ndim != 3:
                    print(f"Warning: {epochs_file} has {epochs.ndim} dimensions, expected 3")
                else:
                    print(f"Verified: {subject}{session} - Epochs shape: {epochs.shape}")
                    
                if labels.ndim != 1:
                    print(f"Warning: {labels_file} has {labels.ndim} dimensions, expected 1")
                else:
                    print(f"Verified: {subject}{session} - Labels shape: {labels.shape}")
                    
                # Check for consistency between epochs and labels
                if epochs.shape[0] != labels.shape[0]:
                    print(f"Error: Shape mismatch between epochs and labels for {subject}{session}")
                else:
                    verified_files += 2
                    
            except Exception as e:
                print(f"Error loading {subject}{session} files: {e}")
    
    print(f"\nVerification complete: {verified_files}/{total_files} files verified successfully.")
    return verified_files == total_files

if __name__ == "__main__":
    success = verify_preprocessed_data()
    if success:
        print("\nAll preprocessed data files are verified and ready for use!")
    else:
        print("\nSome issues found with the preprocessed data files.")