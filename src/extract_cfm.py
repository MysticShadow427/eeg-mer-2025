from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import mne
import os

def load_eeg_data(file_path):
    """Load EEG data from a .fif file."""
    raw = mne.io.read_raw_fif(file_path, preload=True)
    data = raw.get_data()  # Shapm6m-6m
#e: (n_channels, n_times)
    return data

def load_eeg_npy_data(file_path):
    data = np.load(file_path)
    return data

def compute_pcc_cfm(data):
    
    pcc_matrix = np.corrcoef(data)
    return pcc_matrix

def compute_mutual_information(x, y, bins=10):
    """
    Compute the mutual information between two continuous variables by discretizing them.
    
    Parameters:
    x (np.ndarray): 1D array for the first variable.
    y (np.ndarray): 1D array for the second variable.
    bins (int): Number of bins for discretizing continuous data.
    
    Returns:
    float: Mutual information between x and y.
    """
    # Discretize the continuous data
    x_discretized = np.digitize(x, bins=np.histogram_bin_edges(x, bins=bins))
    y_discretized = np.digitize(y, bins=np.histogram_bin_edges(y, bins=bins))
    
    # Compute mutual information
    return mutual_info_score(x_discretized, y_discretized)

def compute_mi_pair(i, j, data, bins=10):
    """
    Helper function to compute mutual information for a single pair of channels.
    
    Parameters:
    i (int): Index of the first channel.
    j (int): Index of the second channel.
    data (np.ndarray): The EEG data array of shape [32, n_t].
    bins (int): Number of bins for discretizing the data.
    
    Returns:
    tuple: (i, j, mutual_information_value)
    """
    mi_value = compute_mutual_information(data[i, :], data[j, :], bins)
    return i, j, mi_value

def compute_mi_cfm(data, n_jobs=4, bins=10):
    """Compute Mutual Information (MI) CFM using parallel processing."""
    n_channels = data.shape[0]
    mi_matrix = np.zeros((n_channels, n_channels))

    # Use parallel processing to compute MI for each pair (i, j)
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_mi_pair)(i, j, data, bins) for i in range(n_channels) for j in range(i + 1, n_channels)
    )

    # Populate the mi_matrix with results
    for i, j, mi_value in results:
        mi_matrix[i, j] = mi_matrix[j, i] = mi_value

    return mi_matrix

# def compute_mi_cfm(data, n_jobs=4):
#     """Compute Mutual Information (MI) CFM using mutual_info_regression with parallel processing."""
#     n_channels = data.shape[0]
#     mi_matrix = np.zeros((n_channels, n_channels))

#     # Define a helper function to compute mutual information for a single pair
#     def compute_mi_pair(i, j):
#         mi_value = mutual_info_regression(data[i, :].reshape(-1, 1), data[j, :])[0]
#         return i, j, mi_value

#     # Use parallel processing to compute MI for each pair (i, j)
#     results = Parallel(n_jobs=4)(
#         delayed(compute_mi_pair)(i, j) for i in range(n_channels) for j in range(i + 1, n_channels)
#     )

#     # Populate the mi_matrix with results
#     for i, j, mi_value in results:
#         mi_matrix[i, j] = mi_matrix[j, i] = mi_value

#     return mi_matrix

def fuse_cfms(cfm1, cfm2):
    """Fuse two CFMs by combining the upper triangle of cfm1 and lower triangle of cfm2."""
    # Initialize the fused matrix directly using cfm1 as a template
    fused_cfm = np.zeros_like(cfm1)

    # Precompute indices once
    upper_indices = np.triu_indices_from(cfm1, k=1)
    lower_indices = np.tril_indices_from(cfm2, k=-1)

    # Fill the upper triangle from cfm1
    fused_cfm[upper_indices] = cfm1[upper_indices]

    # Fill the lower triangle from cfm2
    fused_cfm[lower_indices] = cfm2[lower_indices]

    return fused_cfm

def process_all_files(data_dir,output_dir):
    """Process all .npy files in the directory and save the fused CFMs with the same name but as .npy files."""
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    for file_name in tqdm(files, desc="Processing EEG files", unit="file"):
        file_path = os.path.join(data_dir, file_name)
        data = load_eeg_npy_data(file_path)
        # print(f'[DATA]{data}')
        print(f'[DATA] {data.shape}')
        
        # Compute CFMs
        pcc_cfm = compute_pcc_cfm(data)
        mi_cfm = compute_mi_cfm(data)

        # Fuse PCC and PLV CFMs
        fused_cfm = fuse_cfms(pcc_cfm, mi_cfm)
        print(f'[FUSED] {fused_cfm.shape}')
        print(f'[FUSED DATA] {fused_cfm}')

        # Save the fused CFM with the same name as the original .fif file but with .npy extension
        output_file_path = os.path.join(output_dir, file_name.replace('.npy', '.npy'))
        np.save(output_file_path, fused_cfm)
        print(f"Fused CFM saved to {output_file_path}")

    print("All fused CFMs have been processed and saved.")

# def process_all_files(data_dir,output_file='fused_features.npy'):
#     """Process all .fif files in the directory and save the fused CFMs."""
#     all_fused_cfms = []
#     files = [f for f in os.listdir(data_dir) if f.endswith('.fif')]

#     for file_name in tqdm(files, desc="Processing EEG files", unit="file"):
#         if file_name.endswith('.fif'):
#             file_path = os.path.join(data_dir, file_name)
#             data = load_eeg_data(file_path)
#             print(f'[DATA] {data.shape}')

#             # Compute CFMs
#             pcc_cfm = compute_pcc_cfm(data)
#             # print(pcc_cfm.shape)
#             mi_cfm = compute_mi_cfm(data)
#             # print(mi_cfm.shape)
            
#             # Fuse PCC and PLV CFMs
#             fused_cfm = fuse_cfms(pcc_cfm, mi_cfm)
#             print(f'[FUSED]{fused_cfm.shape}')
#             all_fused_cfms.append(fused_cfm)

#     # Save all fused CFMs to a .npy file
#     np.save(output_file, np.array(all_fused_cfms))
#     # print(all_fused_cfms.shape)
#     print(f"Fused features saved to {output_file}")


if __name__ == '__main__':
    dds = ['/scratch/dkayande/eeg-mer/dataset/eremus_dataset/preprocessed/train','/scratch/dkayande/eeg-mer/dataset/eremus_dataset/preprocessed/test_trial','/scratch/dkayande/eeg-mer/dataset/eremus_dataset/preprocessed/test_subject']
    dos = ['/scratch/dkayande/eeg-mer/dataset/eremus_dataset/cfm/train','/scratch/dkayande/eeg-mer/dataset/eremus_dataset/cfm/test_trial','/scratch/dkayande/eeg-mer/dataset/eremus_dataset/cfm/test_subject']
    
    for i in range(3):
        process_all_files(data_dir=dds[i],output_dir=dos[i])