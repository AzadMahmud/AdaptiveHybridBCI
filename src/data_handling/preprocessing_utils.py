"""
Utility functions and classes for EEG data preprocessing.

This module contains the core components for loading, filtering, epoching,
and preparing BCI competition data for analysis.
"""

import os
import numpy as np
import mne
from scipy import signal
import warnings
from ..preprocessing import artifact_removal

warnings.filterwarnings('ignore')

# Define EEG channels (standard 22 EEG channels for BCI Competition IV 2a)
EEG_CHANNELS = [
    'EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5',
    'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9',
    'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16'
]

def load_gdf_file(filepath):
    """Load GDF file with proper scaling for BCI IV 2a"""
    raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)
    raw.apply_function(lambda x: x * 1e6)  # Convert to microvolts
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
    """
    
    def __init__(self, l_freq=8, h_freq=30, notch_freq=50, sfreq=250):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.sfreq = sfreq

    def apply_bandpass_filter(self, raw):
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, method='iir', 
                   iir_params={'order': 4, 'ftype': 'butter'}, verbose=False)
        return raw
    
    def apply_notch_filter(self, raw):
        raw.notch_filter(freqs=self.notch_freq, verbose=False)
        return raw
    
    def epoch_data(self, raw, trial_starts, labels, tmin=0.5, tmax=2.5):
        smin = int(tmin * self.sfreq)
        smax = int(tmax * self.sfreq)
        n_samples = smax - smin

        available_channels = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
        if not available_channels:
            raise ValueError("No EEG channels found in the raw data!")
        
        eeg_indices = [raw.ch_names.index(ch) for ch in available_channels]
        data = raw.get_data()[eeg_indices, :]

        epochs_data, valid_labels = [], []
        for start_sample, label in zip(trial_starts, labels):
            epoch_start = start_sample + smin
            epoch_end = start_sample + smax

            if epoch_end > data.shape[1]:
                continue

            epoch = data[:, epoch_start:epoch_end]
            if epoch.shape[1] < n_samples:
                pad_width = n_samples - epoch.shape[1]
                epoch = np.pad(epoch, ((0, 0), (0, pad_width)), mode='constant')
            
            epochs_data.append(epoch)
            valid_labels.append(label)
        
        return np.array(epochs_data), np.array(valid_labels)
    
    def preprocess_subject(self, filepath, tmin=0.5, tmax=2.5):
        raw, events, event_dict = load_gdf_file(filepath)
        
        reverse_mapping = {v: k for k, v in event_dict.items()}
        trial_start_code = next((mne_code for mne_code, orig_code in reverse_mapping.items() if '768' in orig_code), None)
        mi_codes = {mne_code: int(orig_code) - 768 for mne_code, orig_code in reverse_mapping.items() if orig_code in ['769', '770', '771', '772']}

        if trial_start_code is None or not mi_codes:
            return np.array([]), np.array([])

        trial_starts, labels = [], []
        for i, event in enumerate(events):
            if event[2] == trial_start_code and i + 1 < len(events) and events[i+1][2] in mi_codes:
                trial_starts.append(event[0])
                labels.append(mi_codes[events[i+1][2]])

        if not labels:
            return np.array([]), np.array([])

        raw = self.apply_bandpass_filter(raw)
        raw = self.apply_notch_filter(raw)
        
        return self.epoch_data(raw, np.array(trial_starts), np.array(labels), tmin, tmax)

def run_preprocessing_pipeline(subject_id, session, data_dir, output_dir, 
                               apply_hwt=False, apply_ssa=False, 
                               tmin=0.5, tmax=2.5):
    """
    Run the complete preprocessing pipeline for a single subject/session.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    preprocessor = BCIPreprocessor()
    filepath = os.path.join(data_dir, f"{subject_id}{session}.gdf")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found: {filepath}")
    
    epochs_data, labels = preprocessor.preprocess_subject(filepath, tmin, tmax)
    
    if epochs_data.size == 0:
        print(f"No epochs extracted for {subject_id}{session}.")
    else:
        if apply_hwt:
            epochs_data = artifact_removal.apply_hwt_to_epochs(epochs_data)
        if apply_ssa:
            epochs_data = artifact_removal.apply_ssa_to_epochs(epochs_data)
    
    np.save(os.path.join(output_dir, f"{subject_id}{session}_epochs.npy"), epochs_data)
    np.save(os.path.join(output_dir, f"{subject_id}{session}_labels.npy"), labels)
    
    print(f"Finished preprocessing for {subject_id}{session}. Data saved in {output_dir}.")
