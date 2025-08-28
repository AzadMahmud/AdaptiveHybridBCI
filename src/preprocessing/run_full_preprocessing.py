"""
Main preprocessing script to run the pipeline for all subjects in the BCI Competition IV Dataset 2a.

This script iterates through all subjects and sessions, applying the preprocessing pipeline
with advanced artifact removal techniques as specified in the Adaptive Hybrid BCI proposal.

Usage:
    cd src/preprocessing
    python run_full_preprocessing.py
"""

import os
import subprocess
import sys

# Add the project root to the Python path so we can import from src.prepocessing
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.preprocessing.preprocess_pipeline import run_preprocessing_pipeline

# Configuration
PROJECT_ROOT = project_root
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'preprocessed')
SCRIPT_PATH = os.path.join(PROJECT_ROOT, 'src', 'preprocessing', 'preprocess_pipeline.py')

# BCI Competition IV Dataset 2a subjects
SUBJECTS = [f"A{str(i).zfill(2)}" for i in range(1, 10)]  # A01 to A09
SESSIONS = ['T', 'E']  # Training and Evaluation

def run_preprocessing_cli(subject_id, session, apply_hwt=True, apply_ssa=False):
    """
    Run preprocessing for a single subject/session using the CLI script.
    """
    cmd = [
        'python', SCRIPT_PATH,
        '--subject_id', subject_id,
        '--session', session,
        '--data_dir', DATA_DIR,
        '--output_dir', OUTPUT_DIR
    ]
    
    if apply_hwt:
        cmd.append('--apply_hwt')
    if apply_ssa:
        cmd.append('--apply_ssa')
        
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Use subprocess.run to capture output and errors
        result = subprocess.run(cmd, check=True, text=True, 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {subject_id}{session}:")
        print(e.stdout)
        print(e.stderr)
        return False
    return True

def run_preprocessing_api(subject_id, session, apply_hwt=True, apply_ssa=False):
    """
    Run preprocessing for a single subject/session using the API.
    """
    try:
        run_preprocessing_pipeline(
            subject_id=subject_id,
            session=session,
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            apply_hwt=apply_hwt,
            apply_ssa=apply_ssa
        )
        return True
    except Exception as e:
        print(f"Error processing {subject_id}{session}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Starting full preprocessing pipeline for BCI Competition IV Dataset 2a...")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Subjects: {SUBJECTS}")
    print(f"Sessions: {SESSIONS}")
    print("-" * 60)
    
    # Check if data directory exists and has files
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found: {DATA_DIR}")
        return
    
    gdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.gdf')]
    if not gdf_files:
        print(f"Error: No .gdf files found in {DATA_DIR}")
        return
        
    print(f"Found {len(gdf_files)} .gdf files in data directory.")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each subject and session
    success_count = 0
    total_count = len(SUBJECTS) * len(SESSIONS)
    
    for subject_id in SUBJECTS:
        for session in SESSIONS:
            print(f"\n--- Processing {subject_id}{session} ---")
            
            # Try using the API first (more direct)
            if run_preprocessing_api(subject_id, session, apply_hwt=True, apply_ssa=False):
                success_count += 1
                print(f"Successfully processed {subject_id}{session}")
            else:
                print(f"Failed to process {subject_id}{session}")
                
    print("\n" + "=" * 60)
    print("Preprocessing Pipeline Completed")
    print(f"Successful: {success_count}/{total_count}")
    print(f"Output files are saved in: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()