"""
Script to test the TCN model on all subjects.

This script runs a quick training test on all subjects to verify 
that the TCN implementation works correctly with the preprocessed data.
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_all_subjects():
    """
    Test the TCN model on all subjects with a quick training run.
    """
    # BCI Competition IV Dataset 2a subjects
    subjects = [f"A{str(i).zfill(2)}" for i in range(1, 10)]  # A01 to A09
    
    print("Testing TCN model on all subjects...")
    print("=" * 50)
    
    success_count = 0
    total_count = len(subjects)
    
    for subject in subjects:
        print(f"\nTesting subject {subject}...")
        
        # Run a quick training test (2 epochs, small batch size)
        cmd = (f"cd {project_root} && source .venv/bin/activate && "
               f"python3 src/models/train_tcn_updated.py --subject_id {subject} "
               f"--epochs 2 --batch_size 16")
        
        # Use os.system for simplicity in this test script
        result = os.system(cmd)
        
        if result == 0:
            print(f"SUCCESS: Subject {subject} completed successfully")
            success_count += 1
        else:
            print(f"FAILED: Subject {subject} failed")
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Successful subjects: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("All subjects completed successfully!")
        print("The TCN implementation is working correctly with all preprocessed data.")
    else:
        print("Some subjects failed. Please check the error messages above.")

if __name__ == "__main__":
    test_all_subjects()