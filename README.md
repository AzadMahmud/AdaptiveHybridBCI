

# Adaptive Hybrid BCI: Fusing Causal-Temporal Convolutions with Handcrafted Features for Subject-Independent Motor Imagery Classification

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
- [ ] Feature extraction (Riemannian/Handcrafted features)
- [ ] Deep learning model (TCN)
- [ ] Adaptive fusion mechanism
- [ ] Training and evaluation framework
- [ ] Baseline implementations
- [ ] Experimentation and analysis