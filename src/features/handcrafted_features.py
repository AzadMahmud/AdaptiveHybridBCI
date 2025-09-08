"""
Handcrafted feature extraction for the Adaptive Hybrid BCI.

This module implements the handcrafted feature stream, including covariance matrix
estimation, Riemannian Alignment, and Common Spatial Pattern (CSP).
"""

import numpy as np
import pyriemann
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.signal import butter, lfilter

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data, axis=-1)
    return y


def get_covariance_matrices(epochs_data):
    """
    Compute covariance matrices for each epoch.
    
    Args:
        epochs_data (np.ndarray): 3D array of shape (n_epochs, n_channels, n_times).
        
    Returns:
        np.ndarray: 3D array of shape (n_epochs, n_channels, n_channels).
    """
    cov_data = pyriemann.estimation.Covariances().transform(epochs_data)
    return cov_data

from pyriemann.tangentspace import TangentSpace

def riemannian_alignment(cov_data, train_indices, test_indices):
    """
    Perform Riemannian Alignment on covariance matrices.
    
    Args:
        cov_data (np.ndarray): 3D array of covariance matrices.
        train_indices (np.ndarray): Indices of the training data.
        test_indices (np.ndarray): Indices of the test data.
        
    Returns:
        np.ndarray: Aligned covariance matrices.
    """
    ts = TangentSpace(tsupdate=True)
    
    # Fit on training data
    ts.fit(cov_data[train_indices])
    
    # Align test data
    cov_test_aligned = ts.transform(cov_data[test_indices])
    
    return cov_test_aligned

def get_csp_features(cov_matrices_train, cov_matrices_test, labels_train):
    """
    Compute CSP features.
    
    Args:
        epochs_data (np.ndarray): 3D array of epoched EEG data.
        labels (np.ndarray): 1D array of labels.
        train_indices (np.ndarray): Indices of the training data.
        test_indices (np.ndarray): Indices of the test data.
        
    Returns:
        np.ndarray: 2D array of CSP features.
    """
    csp = pyriemann.spatialfilters.CSP(nfilter=16)
    
    # Fit CSP on training data
    csp.fit(cov_matrices_train, labels_train)
    
    # Apply CSP to test data
    csp_features = csp.transform(cov_matrices_test)
    
    return csp_features

def get_log_variance_features(epochs_data):
    """
    Compute log-variance features.
    
    Args:
        epochs_data (np.ndarray): 3D array of epoched EEG data.
        
    Returns:
        np.ndarray: 2D array of log-variance features.
    """
    return np.log(np.var(epochs_data, axis=2))