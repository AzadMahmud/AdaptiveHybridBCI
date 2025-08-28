"""
Test script to verify the preprocessing pipeline is working correctly.

This script runs a quick test on a single subject to ensure all components
are functioning as expected.
"""

import os
import sys
import numpy as np

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.preprocessing.preprocess_pipeline import run_preprocessing_pipeline

def test_preprocessing():
    """Test the preprocessing pipeline with a single subject."""
    print("Running preprocessing test...")
    
    # Test with subject A01 training session
    subject_id = "A01"
    session = "T"
    data_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(project_root, "data", "test_preprocessed")
    
    print(f"Project Root: {project_root}")
    print(f"Data Directory: {data_dir}")
    print(f"Output Directory: {output_dir}")
    
    # Check if data directory exists and has the required file
    required_file = os.path.join(data_dir, f"{subject_id}{session}.gdf")
    if not os.path.exists(required_file):
        print(f"Error: Required data file not found: {required_file}")
        print("Please ensure the BCI Competition IV Dataset 2a files are in the data directory.")
        return False
    
    try:
        # Run preprocessing with HWT
        run_preprocessing_pipeline(
            subject_id=subject_id,
            session=session,
            data_dir=data_dir,
            output_dir=output_dir,
            apply_hwt=True,  # Enable HWT
            apply_ssa=False, # Disable SSA for faster test
            tmin=0.5,
            tmax=2.5
        )
        
        # Check if output files were created
        epochs_file = os.path.join(output_dir, f"{subject_id}{session}_epochs.npy")
        labels_file = os.path.join(output_dir, f"{subject_id}{session}_labels.npy")
        
        if not os.path.exists(epochs_file):
            print(f"Error: Epochs file not created: {epochs_file}")
            return False
            
        if not os.path.exists(labels_file):
            print(f"Error: Labels file not created: {labels_file}")
            return False
            
        # Load and verify output
        epochs_data = np.load(epochs_file)
        labels = np.load(labels_file)
        
        print(f"Test successful!")
        print(f"Epochs shape: {epochs_data.shape}")
        print(f"Labels shape: {labels.shape}")
        if len(labels) > 0:
            print(f"Unique labels: {np.unique(labels)}")
        print(f"Output files saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_preprocessing()
    if success:
        print("\nPreprocessing pipeline test completed successfully.")
    else:
        print("\nPreprocessing pipeline test failed.")
        sys.exit(1)