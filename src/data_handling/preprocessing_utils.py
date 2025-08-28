import numpy as np
import mne
from scipy import signal
import os
import warnings
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
    
    # Apply the correct scaling for BCI Competition IV 2a
    # The data is stored in microvolts but might need scaling
    raw.apply_function(lambda x: x * 1e6)  # Convert to microvolts if needed
    
    events, event_dict = mne.events_from_annotations(raw)
    print(f"Event dictionary mapping: {event_dict}")
    return raw, events, event_dict

def extract_labels_from_events(events, event_mapping=None):
    """Extract motor imagery labels from events"""
    if event_mapping is None:
        event_mapping = {769:1, 770:2, 771:3, 772:4}  # Default BCI IV 2a
    mi_events = events[np.isin(events[:, 2], list(event_mapping.keys()))]
    labels = np.array([event_mapping[e[2]] for e in mi_events])
    trial_starts = mi_events[:, 0]
    return labels, trial_starts

def check_raw_data(filepath):
    """Debug function to check raw data before processing"""
    raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)
    
    print(f"Raw data shape: {raw.get_data().shape}")
    print(f"Data range: {np.min(raw.get_data())} to {np.max(raw.get_data())}")
    print(f"Data mean: {np.mean(raw.get_data())}")
    print(f"Channels: {raw.ch_names}")
    
    return raw

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
        
        print(f" Preprocessor initialized:")
        print(f"   - Bandpass filter: {l_freq}-{h_freq} Hz")
        print(f"   - Notch filter: {notch_freq} Hz")
        print(f"   - Sampling rate: {sfreq} Hz")
    
    def apply_bandpass_filter(self, raw):
        print(f"Applying bandpass filter ({self.l_freq}-{self.h_freq} Hz)...")
        raw_filtered = raw.copy()
        raw_filtered.filter(
            l_freq=self.l_freq, h_freq=self.h_freq,
            method='iir', iir_params={'order': 4, 'ftype': 'butter'},
            verbose=False
        )
        return raw_filtered
    
    def apply_notch_filter(self, raw):
        print(f" Applying notch filter ({self.notch_freq} Hz)...")
        raw_notched = raw.copy()
        raw_notched.notch_filter(freqs=self.notch_freq, verbose=False)
        return raw_notched
    
    def epoch_data(self, raw, trial_starts, labels, tmin=0.5, tmax=2.5):
        print(f" Creating epochs ({tmin}s to {tmax}s post-cue)...")
        smin = int(tmin * self.sfreq)
        smax = int(tmax * self.sfreq)
        n_samples = smax - smin

        # Select only available EEG channels
        available_channels = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
        if not available_channels:
            raise ValueError("No EEG channels found in the raw data!")
        print(f"Using EEG channels: {available_channels}")
        eeg_indices = [raw.ch_names.index(ch) for ch in available_channels]
        data = raw.get_data()[eeg_indices, :]

        epochs_data, valid_labels = [], []

        for i, (start_sample, label) in enumerate(zip(trial_starts, labels)):
            epoch_start = start_sample + smin
            epoch_end = start_sample + smax

            # Handle epochs exceeding data boundaries
            if epoch_start >= data.shape[1]:
                print(f"Trial {i+1} skipped: epoch start beyond data length")
                continue
            if epoch_end > data.shape[1]:
                epoch_end = data.shape[1]  # truncate
                print(f" Trial {i+1} truncated: epoch end adjusted to {epoch_end}")

            epoch = data[:, epoch_start:epoch_end]
            if epoch.shape[1] < n_samples:
                print(f"Trial {i+1} warning: epoch length shorter than expected ({epoch.shape[1]} samples)")
            epochs_data.append(epoch)
            valid_labels.append(label)
        
        epochs_data = np.array(epochs_data)
        valid_labels = np.array(valid_labels)
        
        print(f" Created {len(epochs_data)} epochs:")
        if len(epochs_data) > 0:
            print(f"   - Shape: {epochs_data.shape} [trials, channels, samples]")
        print(f"   - Requested duration: {n_samples/self.sfreq:.2f} seconds")
        
        return epochs_data, valid_labels
    
    def preprocess_subject(self, filepath, tmin=0.5, tmax=2.5):
        print(f"\n{'='*60}")
        print(f" Preprocessing: {os.path.basename(filepath)}")
        print(f"{'='*60}")

        # 1. Load raw data with event mapping
        raw, events, event_dict = load_gdf_file(filepath)
        print(f"Event codes in data: {np.unique(events[:, 2])}")

        # 2. Find the correct mapping for trial start and MI events
        # Reverse the event_dict to see what MNE mapped to what
        reverse_mapping = {v: k for k, v in event_dict.items()}
        print(f"Reverse mapping: {reverse_mapping}")

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

        print(f"Trial start code (768): {trial_start_code}")
        print(f"MI codes mapping: {mi_codes}")

        if trial_start_code is None or not mi_codes:
            print(" Could not find required event codes!")
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
            print(" No motor imagery events found!")
            return np.array([]), np.array([])

        print(f"Extracted {len(labels)} trials:")
        class_names = ['Left hand', 'Right hand', 'Both feet', 'Tongue']
        for class_id in range(1,5):
            count = np.sum(labels == class_id)
            print(f"   - Class {class_id} ({class_names[class_id-1]}): {count} trials")

        # 4. Apply filters
        raw_bp = self.apply_bandpass_filter(raw)
        raw_clean = self.apply_notch_filter(raw_bp)

        # 5. Create epochs
        epochs_data, valid_labels = self.epoch_data(raw_clean, trial_starts, labels, tmin, tmax)

        print(f" Preprocessing completed for {os.path.basename(filepath)}")
        return epochs_data, valid_labels

def assess_data_quality(epochs_data, labels, subject_id):
    """
    Assess the quality of preprocessed data.
    Handles empty or zero-length epoch arrays gracefully.
    """
    print(f"\n Data Quality Assessment - {subject_id}")
    print("-" * 40)

    if epochs_data.size == 0:
        print(" No epochs available for this subject.")
        print(f"Labels array length: {len(labels)}")
        return

    n_trials, n_channels, n_samples = epochs_data.shape
    print(f"Trials: {n_trials}")
    print(f"Channels: {n_channels}")
    print(f"Samples per trial: {n_samples}")
    print(f"Sampling rate: 250 Hz")
    print(f"Trial duration: {n_samples/250:.2f}s")

    # Class distribution
    print("\nClass distribution:")
    class_names = ['Left hand', 'Right hand', 'Both feet', 'Tongue']
    for class_id in range(1, 5):
        count = np.sum(labels == class_id)
        percentage = count / len(labels) * 100 if len(labels) > 0 else 0
        print(f"  Class {class_id} ({class_names[class_id-1]}): {count} ({percentage:.1f}%)")

    # Signal statistics
    mean_amplitude = np.mean(np.abs(epochs_data))
    std_amplitude = np.std(epochs_data)
    max_amplitude = np.max(np.abs(epochs_data))

    print("\nSignal statistics (µV):")
    print(f"  Mean absolute amplitude: {mean_amplitude:.2f}")
    print(f"  Standard deviation: {std_amplitude:.2f}")
    print(f"  Maximum amplitude: {max_amplitude:.2f}")

    # Artifact detection
    artifact_threshold = 100  # µV
    artifact_trials = np.any(np.abs(epochs_data) > artifact_threshold, axis=(1, 2))
    n_artifacts = np.sum(artifact_trials)
    print(f"\nArtifact detection (>{artifact_threshold}µV):")
    print(f"  Trials with potential artifacts: {n_artifacts} ({n_artifacts/n_trials*100:.1f}%)")
    if n_artifacts > 0:
        print("   Consider additional artifact removal")