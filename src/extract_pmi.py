import numpy as np
from sklearn.metrics import mutual_info_score
import os 
from tqdm import tqdm

def entropy(X, bins=10):
    hist, _ = np.histogram(X, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist))

def mutual_information(X, Y, bins=10):
    return entropy(X, bins) + entropy(Y, bins) - entropy(np.vstack((X, Y)), bins)

def normalized_mutual_information(X, Y, bins=10):
    mi = mutual_information(X, Y, bins)
    return mi / (entropy(X, bins) + entropy(Y, bins))

def partial_mutual_information(X, Y, Z, bins=10):
    H_XZ = entropy(np.vstack((X, Z)), bins)
    H_YZ = entropy(np.vstack((Y, Z)), bins)
    H_Z = entropy(Z, bins)
    H_XYZ = entropy(np.vstack((X, Y, Z)), bins)
    return H_XZ + H_YZ - H_Z - H_XYZ

def construct_cfm(eeg_data, method='MI', bins=10):
    n_channels = eeg_data.shape[0]
    if method == 'MI':
        cfm = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                cfm[i, j] = mutual_information(eeg_data[i], eeg_data[j], bins)
                cfm[j, i] = cfm[i, j]  # Symmetry
    elif method == 'NMI':
        cfm = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                cfm[i, j] = normalized_mutual_information(eeg_data[i], eeg_data[j], bins)
                cfm[j, i] = cfm[i, j]
    elif method == 'PMI':
        cfm = np.zeros((n_channels, n_channels, n_channels))
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                for z in range(n_channels):
                    if z != i and z != j:
                        cfm[i, j, z] = partial_mutual_information(eeg_data[i], eeg_data[j], eeg_data[z], bins)
                        cfm[j, i, z] = cfm[i, j, z]
    return cfm

def load_eeg_npy_data(file_path):
    data = np.load(file_path)
    return data

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
        # print(f'[DATA] {data.shape}')
        
        # Compute CFMs
        cfm = construct_cfm(data, method='PMI')

        # Save the fused CFM with the same name as the original .fif file but with .npy extension
        output_file_path = os.path.join(output_dir, file_name.replace('.npy', '.npy'))
        np.save(output_file_path, cfm)
        print(f"Fused CFM saved to {output_file_path}")

    print("All fused CFMs have been processed and saved.")

if __name__ == '__main__':
    dds = ['/scratch/dkayande/eeg-mer/dataset/eremus_dataset/preprocessed/train','/scratch/dkayande/eeg-mer/dataset/eremus_dataset/preprocessed/test_trial','/scratch/dkayande/eeg-mer/dataset/eremus_dataset/preprocessed/test_subject']
    dos = ['/scratch/dkayande/eeg-mer/dataset/eremus_dataset/pmi/train','/scratch/dkayande/eeg-mer/dataset/eremus_dataset/pmi/test_trial','/scratch/dkayande/eeg-mer/dataset/eremus_dataset/pmi/test_subject']
    
    for i in range(3):
        process_all_files(data_dir=dds[i],output_dir=dos[i])