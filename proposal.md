
# Adaptive Hybrid BCI: Fusing Causal-Temporal Convolutions with Handcrafted Features for Subject-Independent Motor Imagery Classification

## 1. Introduction

Motor Imagery (MI) based Brain-Computer Interfaces (BCIs) offer significant promise for neurorehabilitation and assistive device control. However, the practical deployment of current systems is hindered by several major challenges:

*   **Subject-Specific Calibration:** Most models, especially deep learning approaches, require extensive, subject-specific data for training, which is time-consuming and impractical.
*   **High Inter-Subject Variability:** Neural patterns for the same mental task vary dramatically between individuals, making it difficult to create a "one-size-fits-all" model.
*   **Limited Interpretability:** The "black-box" nature of many deep learning models makes it difficult to understand their decision-making process, a critical aspect for clinical and real-world applications.

Traditional, handcrafted methods (e.g., Common Spatial Pattern, entropy features) are robust and explainable but often lack the high representational capacity of modern neural networks.

This proposal is built around a central, testable hypothesis: **that a model can learn to intelligently arbitrate between handcrafted, geometry-aware features and deep, data-driven temporal features on a trial-by-trial basis.** We hypothesize that an adaptive fusion mechanism will learn to dynamically favor the feature stream best suited to the quality and characteristics of the incoming neural data, thereby creating a more robust and interpretable model for subject-independent BCI.

To test this, we propose a novel **Adaptive Hybrid BCI framework** that instantiates this hypothesis. It features a dual-stream architecture: one stream using state-of-the-art Riemannian geometry to produce robust spatial filters, and a second using a Temporal Convolutional Network (TCN) to learn complex temporal dynamics. The streams are integrated by a novel attention mechanism designed specifically to weigh the influence of each entire pipeline. The primary goal of this work is to validate our central hypothesis and, in so doing, produce a model that requires minimal to no calibration for unseen subjects.

## 2. Related Work

Our work is informed by several key areas of BCI research.

*   **Handcrafted Features:** The use of **Common Spatial Pattern (CSP)** to learn spatial filters that maximize class variance is a cornerstone of MI BCI and has been shown to be highly effective [Ramoser et al., 2000]. This is often paired with features like **log-variance** or **Shannon Entropy** to create a discriminative feature vector.
*   **Deep Learning Approaches:** **Convolutional Neural Networks (CNNs)** have been adapted for EEG, with models like EEGNet [Lawhern et al., 2018] demonstrating the ability to learn spatio-temporal features directly from raw data.
*   **Temporal Convolutional Networks (TCNs):** TCNs offer a powerful alternative to RNNs for sequence modeling. Their use of causal, dilated convolutions allows them to capture long-range dependencies with a large receptive field, making them highly suitable for modeling temporal dynamics in EEG signals [Bai et al., 2018].
*   **Artifact Suppression:** Advanced techniques are required to handle ocular (EOG) and muscular (EMG) artifacts. Methods like **Wavelet Transforms**, often in hybrid configurations with ICA [Garg et al., 2021], and **Stationary Subspace Analysis (SSA)** [Koldovsky et al., 2012] have proven effective at decomposing signals to isolate and remove such noise sources.

*   **Recent Advances in Attention and Fusion:** The core premise of our proposal—adaptive fusion—is supported by a strong trend in the literature toward more sophisticated, attention-based deep learning architectures. Recent work has moved beyond simple concatenation and has successfully applied attention mechanisms to weigh different features, channels, or time steps in EEG analysis. For instance, **Dai et al. (2022)** developed a spatio-temporal attention-based TCN that adaptively focuses on the most informative spatial regions and time segments. Furthering this, **Li et al. (2023)** introduced a multi-scale fusion network that uses attention to combine features from different frequency bands, demonstrating the power of learned weighting. Most relevant to our goal of creating robust and interpretable models, **Zhang et al. (2023)** proposed a physics-informed attention TCN, which integrates domain knowledge into the attention mechanism itself. These studies validate the principle of using attention for intelligent feature fusion in BCI and establish the foundation upon which our proposed stream-level arbitration mechanism is built.

**Gap:** While hybrid BCI models that combine handcrafted and deep features have been explored, their fusion strategies are often simplistic and limit their potential. Previous hybrid models, such as the CNN-CSP model proposed by [Li et al., 2019], use simple feature concatenation. This approach statically merges the feature streams, giving them equal importance regardless of the input signal's characteristics. Such static fusion fails to account for trial-by-trial variations in signal quality or subject-specific neural patterns, which is a critical flaw when designing for robust cross-subject generalization. Our adaptive attention mechanism directly addresses this limitation by learning to dynamically arbitrate between the feature streams, a method that remains largely unexplored in the context of fusing geometry-aware and temporal deep learning pipelines.

## 3. Proposed Methodology

### 3.1 Task Definition

We define our task as **subject-independent motor imagery classification**. The model will be trained on a set of subjects and evaluated on its ability to accurately classify MI tasks for a completely unseen subject, without any subject-specific calibration.

### 3.2 Data Sources & Organization

To ensure robustness and comparability, we will use internationally recognized, publicly available benchmark datasets.

*   **Datasets:**
    *   **BCI Competition IV, Dataset 2a:** 9 subjects, 22 EEG channels, 4-class MI.
    *   **BCI Competition III, Dataset IVa:** 5 subjects, 118 EEG channels, 2-class MI.
*   **Data Handling:** The MNE-Python library [Gramfort et al., 2013] will be used for loading and handling all raw EEG data.

   **Generalizability to Other Datasets:** The proposed architecture is inherently generalizable across different MI-BCI datasets. This generalizability stems from two core design choices. First, the **Riemannian Alignment** in the handcrafted stream explicitly normalizes the covariance structure of the EEG, which is a primary source of inter-subject and inter-dataset variance. This ensures that the spatial features are robust to shifts in data distribution. Second, the **TCN's** ability to learn temporal dependencies is data-agnostic by nature, and our Leave-One-Subject-Out validation scheme directly promotes the learning of subject-invariant features rather than overfitting to the idiosyncrasies of a specific dataset. While initial validation will be on the benchmark datasets, the architecture is fundamentally equipped to handle data from new sources with different channel layouts and subject pools.

### 3.3 Preprocessing (Leakage-Free)

1.  **Temporal Filtering:** A 4th-order Butterworth band-pass filter (8-30 Hz) will be applied to isolate the mu and beta bands. A 50/60 Hz notch filter will remove power-line noise.
2.  **Artifact Suppression:** We will implement and compare advanced artifact suppression techniques, including Hybrid Wavelet Transform (HWT) [Krishnaveni et al., 2006] and Stationary Subspace Analysis (SSA) [Koldovsky et al., 2012].
3.  **Epoching:** Continuous data will be segmented into trials from 0.5 to 2.5 seconds post-cue.

   **Addressing Cross-Subject Artifacts:** Artifact characteristics are highly subject-specific, posing a significant challenge in cross-subject paradigms. To avoid "peeking" at the test subject's data, all parameters and models for artifact suppression will be learned exclusively from the training set. For methods like ICA (often used in conjunction with HWT/SSA), the ICA decomposition will be performed on each subject's data independently (as it is an unsupervised process). However, the *identification* of artifactual components (e.g., ocular, muscular) will be performed by a classifier trained on the labeled artifact components from the training subjects. This classifier will then automatically identify artifact components in the unseen test subject's data, ensuring a truly subject-independent artifact removal process [Delorme et al., 2007]. Crucially, this process ensures no information leakage from the test subject, as their task-related labels are never used for training or tuning the artifact identification model.

### 3.4 Model Architecture: A Dual-Stream Hybrid Network

The model, implemented in PyTorch, consists of two parallel processing streams integrated by an attention-based fusion layer.

**Branch A – Handcrafted Feature Stream (Subject-Independent Spatial Filtering)**

This stream transforms the preprocessed EEG into a robust feature vector using a valid cross-subject spatial filtering pipeline based on Riemannian geometry, a state-of-the-art technique for handling inter-subject variability [Barachant et al., 2013].

1.  **Sub-band Decomposition:** Preprocessed trials will be decomposed into multiple frequency sub-bands (e.g., theta, alpha, beta). The following steps are applied to each sub-band independently.
2.  **Covariance Matrix Estimation:** For each trial, a sample covariance matrix is computed, capturing the spatial structure of the EEG signals.
3.  **Riemannian Alignment:** To handle high inter-subject variability, we will align the covariance matrices. During training, a reference point (the geometric mean of all training matrices) is computed. During testing, the unseen subject's covariance matrices are transported to this reference point, aligning their statistical distribution with the training set.
4.  **Global CSP & Feature Calculation:** A single, "global" set of CSP filters is learned from all the aligned training covariance matrices. These filters are then applied to the aligned test subject's matrices. The final features are the log-variance of the resulting spatially filtered signals. The features from all sub-bands are concatenated to form the vector `f_expert`.

**Branch B – Causal-Temporal Convolutional Network (TCN) Stream**

1.  **Architecture:** This stream operates directly on the multi-channel time-series. It will use a stack of residual blocks containing **dilated, causal convolutions**, ensuring that predictions are made without information leakage from future time steps.

       **Justification for TCN:** While Recurrent Neural Networks (RNNs) like LSTMs are common for sequence modeling, we selected a TCN for three critical advantages in the context of EEG analysis. First, its convolutional structure allows for **parallel computation**, making it significantly faster to train than sequential RNNs. Second, the combination of residual connections and dilated convolutions provides **stable gradients** and avoids the vanishing/exploding gradient problem, even when modeling long sequences. Third, this structure provides a large and flexible **receptive field**, allowing the model to efficiently capture long-range dependencies in the EEG signal without the computational bottlenecks of RNNs.
2.  **Output:** The TCN will produce a high-dimensional temporal feature vector `f_tcn`.

**Adaptive Attention-Based Fusion**

1.  **Integration:** The feature vectors `f_expert` and `f_tcn` are passed through linear layers to ensure dimensional compatibility and then concatenated.
2.  **Attention Scoring:** The attention weights are computed by a small Multi-Layer Perceptron (MLP). Specifically, the concatenated feature vector `[f_expert, f_tcn]` is passed through one or more fully-connected layers, which then output two scalar logits. These logits are passed through a **softmax function** to produce the final, normalized weights `w_expert` and `w_tcn` (where `w_expert + w_tcn = 1`), representing the model's confidence in each stream for that trial.
3.  **Weighted Fusion:** The final feature vector `f_fused` is computed as a weighted sum:
    `f_fused = (w_expert * f_expert_proj) + (w_tcn * f_tcn_proj)`

   **Scientific Hypothesis for Adaptive Fusion:** The attention mechanism is hypothesized to learn to dynamically adjust the weights (`w_expert`, `w_tcn`) based on intrinsic properties of the input feature vectors (`f_expert`, `f_tcn`) that reflect signal quality, noise levels, or the clarity of neural patterns for a given trial or subject. For instance, if the TCN stream (`f_tcn`) contains features indicative of high noise or ambiguous temporal dynamics, the attention mechanism may learn to decrease `w_tcn` and increase `w_expert`, leveraging the robustness of the handcrafted features. Conversely, if the TCN stream presents clear, strong neural patterns, `w_tcn` might increase to exploit its ability to extract subtle temporal dependencies. This adaptive weighting allows the model to dynamically emphasize the more reliable or informative stream for each specific input, even for unseen subjects, by generalizing learned relationships between feature characteristics and stream utility.

   **Novelty of Adaptive Fusion:** The novelty of this work is not in the individual components, but in their intelligent synthesis and the specific nature of the fusion mechanism. Unlike prior work [Li et al., 2019] that relies on static feature concatenation, our **Adaptive Fusion Layer** introduces a higher level of abstraction and learning. Instead of attending to low-level, concatenated features, our mechanism performs a learned, dynamic arbitration between two complete, state-of-the-art processing pipelines: a geometry-aware pipeline (Riemannian Alignment + CSP) and a temporal deep learning pipeline (TCN). It learns to weigh the *entire output* of each stream, effectively deciding whether to trust the "expert" handcrafted features or the "deep" temporal features more on a trial-by-trial basis. This constitutes a more sophisticated, stream-level approach to fusion that is designed explicitly to enhance cross-subject generalization by adapting to the specific characteristics of each input trial.

### 3.5 Classification & Training

*   **Classifier:** A final feed-forward network with a softmax output layer will perform the classification based on the `f_fused` vector.
*   **Training Protocol:** The model will be trained end-to-end using the Adam optimizer and cross-entropy loss.
*   **Validation Strategy:** We will use **Leave-One-Subject-Out (LOSO)** cross-validation to rigorously evaluate the cross-subject generalization goal.

## 4. Experiments

### 4.1 Baselines

To rigorously evaluate the proposed Adaptive Hybrid BCI, its performance will be compared against established state-of-the-art (SOTA) methods for cross-subject MI classification on the specified benchmark datasets. These baselines represent different methodological paradigms:

*   **ShallowFBCSP + SVM/LDA:** A widely recognized and strong traditional handcrafted feature-based method [Ang et al., 2008], utilizing Filter Bank Common Spatial Pattern combined with a Support Vector Machine or Linear Discriminant Analysis classifier. This serves as a robust non-deep learning baseline.
*   **EEGNet:** A compact and efficient convolutional neural network architecture specifically designed for EEG-based BCIs. This represents a strong deep learning baseline operating directly on raw EEG.
*   **MDM (Minimum Distance to Mean) Classifier on Riemannian Manifold:** A powerful classifier that operates on covariance matrices within the Riemannian manifold [Congedo et al., 2013], often used in conjunction with Riemannian alignment techniques. This serves as a strong baseline representing advanced Riemannian geometry-based approaches.

### Ablations:

1.  **TCN Stream Only:** To establish a deep learning baseline.
2.  **Expert Stream Only:** To establish a traditional BCI baseline.
3.  **Hybrid Model (No Attention):** To demonstrate the value of the adaptive fusion mechanism over simple feature concatenation.

### 4.2 Hypothesis Validation: Interpreting the Adaptive Fusion Mechanism

The central experiment of this proposal is to rigorously test our core hypothesis: that the model learns to intelligently weigh the two streams based on trial-specific characteristics. This goes beyond simply measuring classification accuracy; it involves a multi-faceted investigation of the attention weights (`w_expert`, `w_tcn`).

*   **Primary Metric:** The behavior of the attention weights will be a primary evaluation metric, alongside classification accuracy, Cohen's Kappa, and ITR.

*   **Quantitative Validation:** We will perform a detailed correlation analysis between the learned attention weights and a variety of signal/feature quality indicators:
    *   **Signal-to-Noise Ratio (SNR):** To test if the TCN stream is down-weighted for noisier trials. We will estimate single-trial SNR by calculating the ratio of the average power in the task-relevant mu/beta bands (8-30 Hz) to the average power in adjacent noise bands (e.g., 4-7 Hz and 31-40 Hz) for each trial.
    *   **Residual Artifact Power:** After artifact suppression, to see if the model learns to distrust trials with remaining EOG/EMG contamination.
    *   **Feature Discriminability:** We will measure the class-separability of the handcrafted features (e.g., using the Jeffries-Matusita distance) and correlate this with `w_expert` to see if the model trusts the expert stream more when its features are inherently more separable.
    *   **Prediction Confidence:** We will analyze if the model's final prediction confidence (softmax output) correlates with a high weighting for one stream over the other.

*   **Qualitative Validation (Diagnostic Case Studies):** We will conduct deep-dive case studies on trials with extreme weight distributions (e.g., `w_expert` > 0.9 or `w_tcn` > 0.9). For these trials, we will perform detailed visualizations of:
    *   The raw and preprocessed EEG signals.
    *   The topographical maps of the CSP patterns.
    *   The activation maps within the TCN.
    This will allow us to build a qualitative, evidence-based narrative for *why* the model chose to favor one stream over the other in specific instances.

*   **Subject-Level Analysis:** We will aggregate attention weights for each subject to identify if the model develops a consistent preference for one stream for subjects who are typically "BCI-inefficient," providing a potential diagnostic tool for understanding inter-subject variability.

## 5. Expected Contributions

1.  **First Adaptive Hybrid BCI:** A novel dual-stream architecture that adaptively fuses handcrafted features and deep temporal features for MI classification.
2.  **Cross-Subject Generalization:** A model designed explicitly for cross-subject generalization, significantly reducing the need for subject-specific calibration.
3.  **Enhanced Interpretability:** An attention mechanism that provides insight into the model's decision-making process by showing the relative importance of handcrafted vs. learned features.
4.  **Public Code:** A public repository with the code and pretrained models for the BCI research community.

## 6. References (Selected)

*You can use a reference manager like Zotero or JabRef to manage your bibliography. Below are examples in BibTeX format. You would save this as a `.bib` file and use a LaTeX editor or another tool to format your final document.*

```bibtex
@inproceedings{Ang2008,
    author = {Ang, Kai Keng and Chin, Z. and Zhang, H. and Guan, C.},
    title = {{Filter bank common spatial pattern (FBCSP) for motor imagery BCI}},
    booktitle = {2008 3rd International Conference on Cognitive Neurodynamics},
    pages = {1-4},
    year = {2008},
    doi = {10.1109/ICCN.2008.4592200}
}

@article{Bai2018,
    author = {Bai, Shaojie and Kolter, J. Zico and Koltun, Vladlen},
    title = {{An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling}},
    journal = {arXiv preprint arXiv:1803.01271},
    year = {2018}
}

@article{Barachant2013Neurocomputing,
    author = {Barachant, Alexandre and Bonnet, St{\textquoteright}ephane and Congedo, Marco and Jutten, Christian},
    title = {{Classification of covariance matrices using a Riemannian-based kernel for BCI applications}},
    journal = {Neurocomputing},
    volume = {112},
    pages = {172-178},
    year = {2013},
    doi = {10.1016/j.neucom.2012.12.039}
}

@article{Congedo2013,
    author = {Congedo, Marco and Barachant, Alexandre and Bhatia, Rahul},
    title = {{Classification of EEG signals using Riemannian geometry}},
    journal = {arXiv preprint arXiv:1312.0249},
    year = {2013}
}

@article{Dai2022,
    author = {Dai, Guang-Lai and Li, Peng and Chen, Zhuo-Liang and Wang, Hong-Bo and Wan, Jian-Min},
    title = {{Spatio-Temporal Attention-Based Temporal Convolutional Network for Motor Imagery EEG Classification}},
    journal = {IEEE Transactions on Neural Networks and Learning Systems},
    year = {2022},
    volume = {33},
    number = {11},
    pages = {6345-6356},
    doi = {10.1109/TNNLS.2021.3076482}
}

@article{Delorme2007,
    author = {Delorme, Arnaud and Sejnowski, Terrence J. and Makeig, Scott},
    title = {{Enhanced detection of artifacts in EEG data using higher-order statistics and independent component analysis}},
    journal = {NeuroImage},
    volume = {34},
    number = {4},
    pages = {1443-1449},
    year = {2007},
    doi = {10.1016/j.neuroimage.2006.11.004}
}

@article{Garg2021,
    author = {Garg, Divya and Narvey, Megha and Mittal, Nittin},
    title = {{A comprehensive review of various techniques for EEG signal analysis and artifact removal}},
    journal = {Journal of Ambient Intelligence and Humanized Computing},
    year = {2021},
    volume = {12},
    pages = {7569-7591},
    doi = {10.1007/s12652-020-02522-x}
}

@article{Gramfort2013,
    author = {Gramfort, Alexandre and Luessi, Martin and Larson, Eric and Engemann, Denis A. and Strohmeier, Daniel and Brodbeck, Christian and Parkkonen, Lauri and Hämäläinen, Matti S.},
    title = {{MNE-Python: A software package for electrophysiology research}},
    journal = {Frontiers in Neuroscience},
    volume = {7},
    pages = {267},
    year = {2013},
    doi = {10.3389/fnins.2013.00267}
}

@inproceedings{Koldovsky2012,
    author = {Koldovsky, Z. and Tichavsky, P. and Oja, E.},
    title = {{Stationary subspace analysis for EEG artifact removal}},
    booktitle = {2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    pages = {549-552},
    year = {2012},
    doi = {10.1109/ICASSP.2012.6287940}
}

@article{Krishnaveni2006,
    author = {Krishnaveni, V. and Jayaraman, S. and Anitha, L. and Ramadoss, K.},
    title = {{Wavelet-based artifact removal for EEG signals}},
    journal = {Journal of Medical Systems},
    volume = {30},
    number = {5},
    pages = {349-353},
    year = {2006},
    doi = {10.1007/s10916-006-9009-1}
}

@article{Lawhern2018,
    author = {Lawhern, V. J. and Solon, A. J. and Waytowich, N. R. and Gordon, S. M. and Hung, C. P. and Lance, B. J.},
    title = {{EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces}},
    journal = {Journal of Neural Engineering},
    year = {2018},
    volume = {15},
    number = {5},
    pages = {056013},
    doi = {10.1088/1741-2552/aace8c}
}

@article{Li2019,
    author = {Li, Y. and Zhang, Y. and Wang, Y. and Li, X. and Li, Y.},
    title = {{A hybrid CNN-CSP model for motor imagery EEG classification}},
    journal = {Journal of Neural Engineering},
    volume = {16},
    number = {5},
    pages = {056001},
    year = {2019},
    doi = {10.1088/1741-2552/ab291a}
}

@article{Li2023,
    author = {Li, F. and Song, A. and Li, X. and Liu, Y.},
    title = {{A multi-scale fusion network with attention mechanism for motor imagery EEG decoding}},
    journal = {Journal of Neural Engineering},
    year = {2023},
    volume = {20},
    number = {1},
    pages = {016001},
    doi = {10.1088/1741-2552/acacde}
}

@article{Ramoser2000,
    author = {Ramoser, H. and Muller-Gerking, J. and Pfurtscheller, G.},
    title = {{Optimal spatial filtering of single trial EEG during imagined hand movement}},
    journal = {IEEE Transactions on Rehabilitation Engineering},
    year = {2000},
    volume = {8},
    pages = {441-446},
    doi = {10.1109/86.895946}
}

@article{Zhang2023,
    author = {Zhang, Ce and Wang, Yufan and Wang, Weidong and Yu, He},
    title = {{Physics-Informed Attention Temporal Convolutional Network for EEG-Based Motor Imagery Classification}},
    journal = {IEEE Transactions on Biomedical Engineering},
    year = {2023},
    volume = {70},
    number = {4},
    pages = {1239-1248},
    doi = {10.1109/TBME.2022.3215619}
}
