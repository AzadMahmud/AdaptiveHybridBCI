"""
Modular Preprocessing Pipeline for BCI Competition IV Dataset 2a.

This script provides a command-line interface to run the complete preprocessing
pipeline, including advanced artifact removal, as specified in the Adaptive Hybrid BCI proposal.

It loads raw GDF files, applies preprocessing steps, and saves the resulting
epoched data to NumPy files for use in downstream analysis and modeling.

Usage:
    python preprocess_pipeline.py --subject_id A01 --session T --output_dir ../data/preprocessed
"""

import os
import argparse
import numpy as np
import mne
from . import artifact_removal # Import from the same package

# Use the EEG_CHANNELS definition from the existing preprocessing_utils
EEG_CHANNELS = [
    'EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5',
    'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9',
    'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16'
]

def load_gdf_file(filepath):
    """Load GDF file with proper scaling for BCI IV 2a"""
    raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)
    # Apply the correct scaling for BCI Competition IV 2a
    # The data is stored in microvolts but might need scaling
    raw.apply_function(lambda x: x * 1e6)  # Convert to microvolts if needed
    events, event_dict = mne.events_from_annotations(raw)
    return raw, events, event_dict

def extract_labels_from_events(events, event_mapping=None):
    """Extract motor imagery labels from events"""
    if event_mapping is None:
        event_mapping = {769:1, 770:2, 771:3, 772:4}  # Default BCI IV 2a
    mi_events = events[np.isin(events[:, 2], list(event_mapping.keys()))]
    labels = np.array([event_mapping[e[2]] for e in mi_events])
    trial_starts = mi_events[:, 0]
    return labels, trial_starts

class BCIPreprocessor:
    """
    Preprocessing pipeline for BCI Competition IV 2a dataset.
    Automatically selects available EEG channels and handles edge cases.
    """
    
    def __init__(self, l_freq=8, h_freq=30, notch_freq=50, sfreq=250):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.sfreq = sfreq

    def apply_bandpass_filter(self, raw):
        raw_filtered = raw.copy()
        raw_filtered.filter(
            l_freq=self.l_freq, h_freq=self.h_freq,
            method='iir', iir_params={'order': 4, 'ftype': 'butter'},
            verbose=False
        )
        return raw_filtered
    
    def apply_notch_filter(self, raw):
        raw_notched = raw.copy()
        raw_notched.notch_filter(freqs=self.notch_freq, verbose=False)
        return raw_notched
    
    def epoch_data(self, raw, trial_starts, labels, tmin=0.5, tmax=2.5):
        smin = int(tmin * self.sfreq)
        smax = int(tmax * self.sfreq)
        n_samples = smax - smin

        # Select only available EEG channels
        available_channels = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
        if not available_channels:
            raise ValueError("No EEG channels found in the raw data!")
        eeg_indices = [raw.ch_names.index(ch) for ch in available_channels]
        data = raw.get_data()[eeg_indices, :]

        epochs_data, valid_labels = [], []

        for i, (start_sample, label) in enumerate(zip(trial_starts, labels)):
            epoch_start = start_sample + smin
            epoch_end = start_sample + smax

            # Handle epochs exceeding data boundaries
            if epoch_start >= data.shape[1]:
                continue
            if epoch_end > data.shape[1]:
                epoch_end = data.shape[1]  # truncate

            epoch = data[:, epoch_start:epoch_end]
            if epoch.shape[1] < n_samples:
                # Pad if shorter
                pad_width = n_samples - epoch.shape[1]
                epoch = np.pad(epoch, ((0, 0), (0, pad_width)), mode='constant')
            epochs_data.append(epoch)
            valid_labels.append(label)
        
        epochs_data = np.array(epochs_data)
        valid_labels = np.array(valid_labels)
        
        return epochs_data, valid_labels
    
    def preprocess_subject(self, filepath, tmin=0.5, tmax=2.5):
        # 1. Load raw data with event mapping
        raw, events, event_dict = load_gdf_file(filepath)

        # 2. Find the correct mapping for trial start and MI events
        reverse_mapping = {v: k for k, v in event_dict.items()}

        # Find which MNE code corresponds to original codes
        trial_start_code = None
        mi_codes = {}
        
        for mne_code, orig_code_str in reverse_mapping.items():
            try:
                orig_code = int(orig_code_str)
                if orig_code == 768:
                    trial_start_code = mne_code
                elif orig_code in [769, 770, 771, 772]:
                    mi_codes[mne_code] = orig_code - 768  # Map to class 1-4
            except ValueError:
                continue

        if trial_start_code is None or not mi_codes:
            return np.array([]), np.array([])

        # 3. Extract trials
        trial_starts = []
        labels = []

        for i, event in enumerate(events):
            code = event[2]
            if code == trial_start_code and i+1 < len(events):
                next_code = events[i+1][2]
                if next_code in mi_codes:
                    trial_starts.append(event[0])
                    labels.append(mi_codes[next_code])

        trial_starts = np.array(trial_starts)
        labels = np.array(labels)

        if len(labels) == 0:
            return np.array([]), np.array([])

        # 4. Apply filters
        raw_bp = self.apply_bandpass_filter(raw)
        raw_clean = self.apply_notch_filter(raw_bp)

        # 5. Create epochs
        epochs_data, valid_labels = self.epoch_data(raw_clean, trial_starts, labels, tmin, tmax)

        return epochs_data, valid_labels

def run_preprocessing_pipeline(subject_id, session, data_dir, output_dir, 
                               apply_hwt=False, apply_ssa=False, 
                               tmin=0.5, tmax=2.5):
    """
    Run the complete preprocessing pipeline for a single subject/session.
    
    Args:
        subject_id (str): Subject identifier (e.g., 'A01').
        session (str): Session identifier ('T' for training, 'E' for evaluation).
        data_dir (str): Path to the directory containing raw GDF files.
        output_dir (str): Path to the directory where preprocessed data will be saved.
        apply_hwt (bool): Whether to apply Hybrid Wavelet Transform denoising.
        apply_ssa (bool): Whether to apply simplified SSA denoising.
        tmin (float): Start time for epoching, relative to cue.
        tmax (float): End time for epoching, relative to cue.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = BCIPreprocessor()
    
    # Construct file path
    filepath = os.path.join(data_dir, f"{subject_id}{session}.gdf")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found: {filepath}")
    
    print(f"Preprocessing: {filepath}")
    
    # Run basic preprocessing
    epochs_data, labels = preprocessor.preprocess_subject(filepath, tmin, tmax)
    
    if epochs_data.size == 0:
        print("No epochs extracted. Saving empty arrays.")
        np.save(os.path.join(output_dir, f"{subject_id}{session}_epochs.npy"), epochs_data)
        np.save(os.path.join(output_dir, f"{subject_id}{session}_labels.npy"), labels)
        return
    
    print(f"Extracted {len(labels)} epochs of shape {epochs_data.shape}")
    
    # Apply advanced artifact removal techniques
    if apply_hwt:
        print("Applying Hybrid Wavelet Transform...")
        epochs_data = artifact_removal.apply_hwt_to_epochs(epochs_data)
        
    if apply_ssa:
        print("Applying simplified Stationary Subspace Analysis...")
        epochs_data = artifact_removal.apply_ssa_to_epochs(epochs_data)
    
    # Save preprocessed data
    epochs_file = os.path.join(output_dir, f"{subject_id}{session}_epochs.npy")
    labels_file = os.path.join(output_dir, f"{subject_id}{session}_labels.npy")
    
    np.save(epochs_file, epochs_data)
    np.save(labels_file, labels)
    
    print(f"Preprocessed data saved to:")
    print(f"  Epochs: {epochs_file}")
    print(f"  Labels: {labels_file}")
    print("Preprocessing pipeline completed.")

def main():
    parser = argparse.ArgumentParser(description="Preprocess BCI Competition IV Dataset 2a.")
    parser.add_argument('--subject_id', type=str, required=True, help='Subject ID (e.g., A01)')
    parser.add_argument('--session', type=str, choices=['T', 'E'], required=True, 
                        help='Session type: T (training) or E (evaluation)')
    parser.add_argument('--data_dir', type=str, default='../data', 
                        help='Path to the directory containing raw GDF files')
    parser.add_argument('--output_dir', type=str, default='../data/preprocessed', 
                        help='Path to the directory to save preprocessed data')
    parser.add_argument('--apply_hwt', action='store_true', 
                        help='Apply Hybrid Wavelet Transform denoising')
    parser.add_argument('--apply_ssa', action='store_true', 
                        help='Apply simplified Stationary Subspace Analysis')
    parser.add_argument('--tmin', type=float, default=0.5, 
                        help='Start time for epoching (seconds relative to cue)')
    parser.add_argument('--tmax', type=float, default=2.5, 
                        help='End time for epoching (seconds relative to cue)')
    
    args = parser.parse_args()
    
    # Resolve paths relative to the script's location if they are relative
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', args.data_dir))
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', args.output_dir))
        
    run_preprocessing_pipeline(
        subject_id=args.subject_id,
        session=args.session,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        apply_hwt=args.apply_hwt,
        apply_ssa=args.apply_ssa,
        tmin=args.tmin,
        tmax=args.tmax
    )

if __name__ == '__main__':
    main()