# Preprocessing Pipeline

This directory contains the complete preprocessing pipeline for the BCI Competition IV Dataset 2a, as specified in the Adaptive Hybrid BCI proposal.

## Overview

The preprocessing pipeline performs the following steps:

1. Loading raw GDF files
2. Applying bandpass filtering (8-30 Hz)
3. Applying notch filtering (50 Hz)
4. Epoching the data (0.5-2.5 seconds post-cue)
5. Applying advanced artifact removal techniques:
   - Hybrid Wavelet Transform (HWT)
   - Stationary Subspace Analysis (SSA) - simplified implementation
6. Saving the preprocessed data in NumPy format

## Files

- `artifact_removal.py`: Implementation of advanced artifact removal techniques (HWT, SSA)
- `preprocess_pipeline.py`: Main preprocessing pipeline that can be used as a module
- `run_full_preprocessing.py`: Script to run the preprocessing pipeline for all subjects
- `test_preprocessing.py`: Test script to verify the pipeline works correctly
- `verify_preprocessed_data.py`: Script to verify the generated preprocessed data files
- `__init__.py`: Package initialization file

## Usage

### Running the full preprocessing pipeline

```bash
cd src/preprocessing
python run_full_preprocessing.py
```

### Running the preprocessing pipeline for a single subject

```bash
cd src/preprocessing
python preprocess_pipeline.py --subject_id A01 --session T --apply_hwt
```

### Testing the preprocessing pipeline

```bash
cd src/preprocessing
python test_preprocessing.py
```

### Verifying the preprocessed data

```bash
cd src/preprocessing
python verify_preprocessed_data.py
```

## Output

The preprocessed data is saved in the `data/preprocessed` directory as NumPy arrays:

- `{subject}{session}_epochs.npy`: Epoched EEG data with shape (n_epochs, n_channels, n_times)
- `{subject}{session}_labels.npy`: Labels for each epoch with shape (n_epochs,)

For the BCI Competition IV Dataset 2a:
- n_channels = 22 (EEG channels)
- n_times = 500 (0.5-2.5 seconds at 250 Hz)
- Labels are 1-4 corresponding to:
  1. Left hand
  2. Right hand
  3. Both feet
  4. Tongue

## Artifact Removal Techniques

### Hybrid Wavelet Transform (HWT)

The HWT implementation uses wavelet decomposition and thresholding to remove artifacts. It processes each channel independently using the `pywt` library.

### Stationary Subspace Analysis (SSA)

The SSA implementation is a simplified version that uses PCA on time-delay embedded data. Note that a full SSA implementation for multi-channel EEG would be more complex.

## Requirements

- Python 3.x
- MNE-Python
- NumPy
- SciPy
- PyWavelets
- scikit-learn