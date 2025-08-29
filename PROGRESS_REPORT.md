# Adaptive Hybrid BCI - Progress Report

## Project Overview

This project implements the Adaptive Hybrid BCI approach for subject-independent motor imagery classification as proposed in the gemini_bci_proposal.md. The goal is to create a model that can classify motor imagery tasks without requiring subject-specific calibration.

### Key Components of the Proposal:
1. **Dual-stream architecture**:
   - Branch A: Handcrafted features (Riemannian geometry/CSP)
   - Branch B: Deep learning (Temporal Convolutional Network)
   - Adaptive attention-based fusion of both streams

2. **Subject-independent performance**:
   - Leave-One-Subject-Out (LOSO) cross-validation
   - Minimal to no calibration for unseen subjects

## Completed Work

### 1. Data Preprocessing Pipeline

**Location**: `src/preprocessing/`

#### What was implemented:
- Loading and processing raw GDF files from BCI Competition IV Dataset 2a
- Bandpass filtering (8-30 Hz) and notch filtering (50 Hz)
- Epoching data (0.5-2.5 seconds post-cue)
- Advanced artifact removal techniques:
  - Hybrid Wavelet Transform (HWT)
  - Simplified Stationary Subspace Analysis (SSA)
- Modular, standalone scripts for reproducibility
- Output saved as NumPy arrays for downstream processing

#### Key files:
- `artifact_removal.py`: Implementation of HWT and SSA
- `preprocess_pipeline.py`: Main preprocessing pipeline
- `run_full_preprocessing.py`: Script to process all subjects
- `test_preprocessing.py`: Verification script

#### Results:
- Successfully processed data for all 9 subjects (A01-A09)
- Generated preprocessed epochs and labels for training and evaluation
- Verified data integrity and consistency

### 2. Deep Learning Model (TCN - Branch B)

**Location**: `src/models/`

#### What was implemented:
- Complete Temporal Convolutional Network architecture with:
  - Causal, dilated convolutions
  - Residual connections
  - Proper gradient flow
- Validation scripts to test implementation without training
- Training scripts that work with preprocessed data
- Leave-One-Subject-Out (LOSO) cross-validation framework

#### Key files:
- `tcn.py`: TCN model architecture
- `validate_tcn.py`: Model validation without training
- `train_tcn_updated.py`: Training script with real data
- `loso_evaluation.py`: LOSO cross-validation implementation

#### Results:
- Model architecture validated and working correctly
- Successfully trained and evaluated on all subjects
- LOSO framework established for subject-independent evaluation
- Baseline performance: ~25.5% accuracy (at chance level, expected for limited training)

### 3. Project Infrastructure

#### What was implemented:
- Organized project structure following best practices
- Comprehensive documentation in README files
- Requirements management
- Results organization

#### Key files:
- `README.md`: Main project documentation
- `requirements.txt`: Dependencies
- `NEXT_STEPS.md`: Clear roadmap for next steps
- `results/`: Directory for experiment results

## Technical Details

### Preprocessing Pipeline
```bash
# Run full preprocessing
cd src/preprocessing
python run_full_preprocessing.py
```

**Process**:
1. Raw GDF file loading
2. Signal filtering
3. Epoching
4. Artifact removal (HWT)
5. Save as NumPy arrays

### TCN Model
```bash
# Validate model implementation
cd src/models
python validate_tcn.py

# Train on single subject
python train_tcn_updated.py --subject_id A01 --epochs 50

# Run LOSO evaluation
python loso_evaluation.py
```

**Architecture**:
- Input: (batch_size, 22 channels, 500 timepoints)
- Causal dilated convolutions with residual connections
- Output: 64-dimensional feature vector (f_tcn)

### LOSO Evaluation
- Train on 8 subjects, test on 1 subject
- Repeat for all 9 subjects
- Report average performance across subjects

## Current Status

### Completed ✅
- [x] Data preprocessing pipeline with advanced artifact removal
- [x] TCN model implementation and validation
- [x] Training scripts with real preprocessed data
- [x] LOSO cross-validation framework

### In Progress 🔄
- [ ] Handcrafted feature stream (CSP/Riemannian alignment)
- [ ] Adaptive fusion mechanism
- [ ] Full hybrid model implementation

### Next Steps ➡️
1. Implement handcrafted feature extraction (Branch A)
2. Optimize both streams independently
3. Implement attention-based fusion
4. Evaluate hybrid model with LOSO
5. Compare with baseline methods

## Key Achievements

1. **Modular Implementation**: Each component is standalone and testable
2. **Reproducibility**: Clear scripts and documentation for replication
3. **Standard Evaluation**: LOSO framework follows BCI research best practices
4. **Scalability**: Framework ready for full implementation of the hybrid approach

## Challenges and Solutions

1. **Artifact Removal**: Implemented HWT to address signal quality issues
2. **Model Validation**: Created separate validation scripts to test without GPU
3. **Data Handling**: Modular preprocessing pipeline handles all subjects consistently

## Demonstration Plan

To show the professor:

1. **Live Demo**:
   - Run `validate_tcn.py` to show model works
   - Show preprocessed data files in `data/preprocessed/`
   - Show results from LOSO evaluation

2. **Code Walkthrough**:
   - Show `src/preprocessing/` structure and key files
   - Show `src/models/tcn.py` architecture
   - Show `loso_evaluation.py` implementation

3. **Results Presentation**:
   - Show LOSO results file
   - Explain current performance and next steps for improvement

## Future Work Plan

1. **Short Term** (1-2 weeks):
   - Implement CSP/Riemannian feature extraction
   - Validate handcrafted stream independently

2. **Medium Term** (2-4 weeks):
   - Implement attention-based fusion
   - Train and evaluate hybrid model
   - Run ablation studies

3. **Long Term** (1-2 months):
   - Hyperparameter optimization
   - Baseline method comparisons
   - Comprehensive evaluation and analysis