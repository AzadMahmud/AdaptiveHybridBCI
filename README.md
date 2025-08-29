# Adaptive Hybrid BCI

This project implements the Adaptive Hybrid BCI approach proposed in `gemini_bci_proposal.md` for subject-independent motor imagery classification.

## Project Structure

- `data/`: Raw and preprocessed data files
- `src/`: Source code
  - `preprocessing/`: Preprocessing pipeline for BCI Competition IV Dataset 2a
  - `features/`: Feature extraction methods (to be implemented)
  - `models/`: Model architectures and training scripts
  - `utils/`: Utility functions
- `notebooks/`: Jupyter notebooks for exploration and visualization
- `results/`: Experiment results and evaluations
- `scripts/`: Utility scripts

## Preprocessing Pipeline

The preprocessing pipeline has been implemented and can be found in `src/preprocessing/`. It includes:

1. Loading raw GDF files
2. Applying bandpass and notch filtering
3. Epoching the data
4. Applying advanced artifact removal techniques (HWT, SSA)
5. Saving the preprocessed data in NumPy format

To run the full preprocessing pipeline:

```bash
cd src/preprocessing
python run_full_preprocessing.py
```

For more details, see `src/preprocessing/README.md`.

## Deep Learning Model (TCN)

The Temporal Convolutional Network (TCN) model, which is the deep learning component (Branch B) of the Adaptive Hybrid BCI, has been implemented and tested:

1. Complete TCN architecture implementation with causal, dilated convolutions and residual connections
2. Validation script to test the model implementation without training
3. Updated training script that works with the preprocessed data
4. Leave-One-Subject-Out (LOSO) cross-validation framework for subject-independent evaluation

To validate the TCN implementation:

```bash
cd src/models
python validate_tcn.py
```

To train the TCN model on a single subject:

```bash
cd src/models
python train_tcn_updated.py --subject_id A01 --epochs 10 --batch_size 32
```

To run LOSO cross-validation:

```bash
cd src/models
python loso_evaluation.py
```

For more details, see `src/models/README.md`.

## Requirements

All requirements are listed in `requirements.txt`. To install:

```bash
pip install -r requirements.txt
```

Note: The preprocessing pipeline requires the `PyWavelets` package for HWT implementation, which can be installed with:

```bash
pip install PyWavelets
```

## Progress

- [x] Project setup and proposal documentation
- [x] Data handling and basic preprocessing
- [x] Advanced preprocessing pipeline with artifact removal
- [x] Deep learning model (TCN) implementation and validation
- [x] LOSO evaluation framework for subject-independent performance
- [ ] Feature extraction (Riemannian/Handcrafted features)
- [ ] Adaptive fusion mechanism
- [ ] Training and evaluation framework for hybrid model
- [ ] Baseline implementations
- [ ] Experimentation and analysis