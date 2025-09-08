import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.features.handcrafted_features import get_covariance_matrices, get_csp_features, get_log_variance_features, butter_bandpass_filter
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Configuration
DATA_DIR = os.path.join(project_root, 'data', 'preprocessed')
RESULTS_DIR = os.path.join(project_root, 'results', 'loso_handcrafted')
os.makedirs(RESULTS_DIR, exist_ok=True)

# BCI Competition IV Dataset 2a subjects
SUBJECTS = [f"A{str(i).zfill(2)}" for i in range(1, 10)]  # A01 to A09

def load_subject_data(subject_id):
    epochs_path = os.path.join(DATA_DIR, f"{subject_id}T_epochs.npy")
    labels_path = os.path.join(DATA_DIR, f"{subject_id}T_labels.npy")
    
    if not os.path.exists(epochs_path) or not os.path.exists(labels_path):
        return None, None
    
    epochs_data = np.load(epochs_path)
    labels = np.load(labels_path)
    
    if epochs_data.size == 0 or labels.size == 0:
        return None, None
        
    return epochs_data, labels

def loso_cross_validation():
    print("Starting Leave-One-Subject-Out Cross-Validation for Handcrafted Features with LDA")
    print("=" * 60)
    
    results = {}
    
    # Load all data
    all_epochs = []
    all_labels = []
    subject_indices = []
    current_index = 0
    for subject in SUBJECTS:
        epochs_data, labels = load_subject_data(subject)
        if epochs_data is not None:
            all_epochs.append(epochs_data)
            all_labels.append(labels)
            subject_indices.append(np.arange(current_index, current_index + len(labels)))
            current_index += len(labels)
        else:
            subject_indices.append(np.array([]))

    all_epochs = np.concatenate(all_epochs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    for i, test_subject in enumerate(SUBJECTS):
        print(f"\n[{i+1}/{len(SUBJECTS)}] Testing on subject: {test_subject}")
        
        test_indices = subject_indices[i]
        train_indices = np.concatenate([subject_indices[j] for j in range(len(SUBJECTS)) if i != j])

        if len(test_indices) == 0:
            print(f"  Skipping {test_subject} - no data available")
            results[test_subject] = {'accuracy': 0.0, 'kappa': 0.0, 'status': 'no_data'}
            continue

        # Sub-band decomposition
        bands = [(8, 12), (12, 15), (15, 30)]
        handcrafted_features_train_list = []
        handcrafted_features_test_list = []

        for lowcut, highcut in bands:
            # Filter the data
            epochs_band = butter_bandpass_filter(all_epochs, lowcut, highcut, fs=250)

            # Feature Extraction
            cov_matrices = get_covariance_matrices(epochs_band)
            cov_train = cov_matrices[train_indices]
            cov_test = cov_matrices[test_indices]

            ts = TangentSpace(tsupdate=True)
            ts.fit(cov_train)
            cov_test_aligned = ts.transform(cov_test)
            cov_test_aligned_reprojected = ts.inverse_transform(cov_test_aligned)

            csp_features_train = get_csp_features(cov_train, cov_train, all_labels[train_indices])
            csp_features_test = get_csp_features(cov_train, cov_test_aligned_reprojected, all_labels[train_indices])

            log_var_features_train = get_log_variance_features(epochs_band[train_indices])
            log_var_features_test = get_log_variance_features(epochs_band[test_indices])

            handcrafted_features_train = np.concatenate([csp_features_train, log_var_features_train], axis=1)
            handcrafted_features_test = np.concatenate([csp_features_test, log_var_features_test], axis=1)

            handcrafted_features_train_list.append(handcrafted_features_train)
            handcrafted_features_test_list.append(handcrafted_features_test)

        handcrafted_features_train = np.concatenate(handcrafted_features_train_list, axis=1)
        handcrafted_features_test = np.concatenate(handcrafted_features_test_list, axis=1)

        print("Handcrafted features train shape:", handcrafted_features_train.shape)
        print("Handcrafted features test shape:", handcrafted_features_test.shape)

        # Train and evaluate LDA
        lda = LDA()
        lda.fit(handcrafted_features_train, all_labels[train_indices])
        preds = lda.predict(handcrafted_features_test)
        
        accuracy = accuracy_score(all_labels[test_indices], preds)
        kappa = cohen_kappa_score(all_labels[test_indices], preds)
        
        print(f"  Results for {test_subject}:")
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    Kappa: {kappa:.4f}")
        
        results[test_subject] = {'accuracy': accuracy, 'kappa': kappa, 'status': 'success'}
    
    print("\n" + "=" * 60)
    print("LOSO CROSS-VALIDATION RESULTS")
    print("=" * 60)
    
    successful_results = [r for r in results.values() if r['status'] == 'success']
    
    if successful_results:
        accuracies = [r['accuracy'] for r in successful_results]
        kappas = [r['kappa'] for r in successful_results]
        
        print(f"Number of subjects evaluated: {len(successful_results)}")
        print(f"Average Accuracy: {np.mean(accuracies):.4f} \u00b1 {np.std(accuracies):.4f}")
        print(f"Average Kappa: {np.mean(kappas):.4f} \u00b1 {np.std(kappas):.4f}")
        
        print("\nPer-subject results:")
        for subject, result in results.items():
            if result['status'] == 'success':
                print(f"  {subject}: Accuracy = {result['accuracy']:.4f}, Kappa = {result['kappa']:.4f}")
            else:
                print(f"  {subject}: {result['status']}")
                
        results_file = os.path.join(RESULTS_DIR, 'loso_handcrafted_results.txt')
        with open(results_file, 'w') as f:
            f.write("LOSO Cross-Validation Results for Handcrafted Features with LDA\n")
            f.write("=" * 50 + "\n")
            f.write(f"Average Accuracy: {np.mean(accuracies):.4f} \u00b1 {np.std(accuracies):.4f}\n")
            f.write(f"Average Kappa: {np.mean(kappas):.4f} \u00b1 {np.std(kappas):.4f}\n\n")
            f.write("Per-subject results:\n")
            for subject, result in results.items():
                if result['status'] == 'success':
                    f.write(f"  {subject}: Accuracy = {result['accuracy']:.4f}, Kappa = {result['kappa']:.4f}\n")
                else:
                    f.write(f"  {subject}: {result['status']}\n")
                    
        print(f"\nResults saved to: {results_file}")
    else:
        print("No successful evaluations completed.")
        
    return results

if __name__ == '__main__':
    loso_cross_validation()