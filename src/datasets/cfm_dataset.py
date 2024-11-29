# import os
# import mne
# import json
# import numpy as np
# from tqdm import tqdm
# import src.config as config
# from torchvision import transforms as T
# from torch.utils.data import Dataset, DataLoader
# from src.eeg_transforms import RandomCrop, ToTensor, Standardize
# # from src.utils import EEGToMelSpectrogram
# import torchaudio
# import torch
# import torch.nn.functional as F
# from mne_connectivity import SpectralConnectivity
# # from mne.connectivity import spectral_connectivity
# from scipy.stats import pearsonr
# from sklearn.feature_selection import mutual_info_regression
# import pyinform

# mne.set_log_level("ERROR")
 
# class EremusDataset(Dataset):
#     def __init__(self, subdir, split_dir, split="train", task="subject_identification", ext="fif", transform=None, prefix="",image_transform=False):
        
#         self.dataset_dir = config.get_attribute("dataset_path", prefix=prefix)
#         self.subdir = os.path.join(subdir, split) if "test" in split else os.path.join(subdir, "train")
#         self.split_dir = split_dir
#         self.transform = transform
#         self.split = split
#         self.label_name = "subject_id" if task == "subject_identification" else "label"
#         self.ext = ext
        
#         splits = json.load(open(os.path.join(split_dir, f"splits_{task}.json")))
#         self.samples = splits[split]
        
#         files = []
#         for sample in self.samples:
#             #path = os.path.join(self.dataset_dir, self.subdir, sample['filename_preprocessed'])
#             path = os.path.join(self.dataset_dir, self.subdir, f"{sample['id']}_eeg.{self.ext}")
#             files.append(path)
#         files = list(set(files))
#         #self.files = {f: np.load(f)['arr_0'] for f in files}
#         if self.ext == "npy":
#             self.files = {f: np.load(f) for f in tqdm(files)}
#         elif self.ext == "fif":
#             self.files = {f: mne.io.read_raw_fif(f, preload=True).get_data() for f in tqdm(files)}
#         else:
#             raise ValueError(f"Extension {ext} not recognized")
        
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         data = self.files[os.path.join(self.dataset_dir, self.subdir, f"{sample['id']}_eeg.{self.ext}")]

#         # data = np.array(data)
#         # Compute CFMs
#         pcc_cfm = self.compute_pcc_cfm(data)
#         # plv_cfm = self.compute_plv_cfm(data)
#         mi_cfm = self.compute_mi_cfm(data)
#         # te_cfm = self.compute_te_cfm(data)

#          # Fuse CFMs
#         fused_cfm = self.fuse_cfms(mi_cfm, pcc_cfm)


#         sample = {
#             "id": sample['id'],
#             "eeg": data,
#             "label": sample[self.label_name] if "test" not in self.split else -1,
#         }
#         if self.transform:
#             sample = self.transform(sample)
        
#         sample['cfm'] = torch.tensor(fused_cfm, dtype=torch.float32)
        
#         return sample   

#     def compute_pcc_cfm(self, data):
#         """Compute Pearson Correlation Coefficient (PCC) CFM."""
#         n_channels = data.shape[0]
#         pcc_matrix = np.zeros((n_channels, n_channels))
#         for i in range(n_channels):
#             for j in range(i + 1, n_channels):
#                 pcc, _ = pearsonr(data[i, :], data[j, :])
#                 pcc_matrix[i, j] = pcc_matrix[j, i] = pcc
#         return pcc_matrix

#     def compute_plv_cfm(self, data):
#         """Compute Phase Locking Value (PLV) CFM."""
#         data_reshaped = data[np.newaxis, :, :]  # Shape (1, n_channels, n_times)
#         plv_matrix, _, _, _, _ = SpectralConnectivity(
#             data_reshaped, method='plv', sfreq=128, fmin=8, fmax=12, faverage=True, verbose=False
#         )
#         return np.squeeze(plv_matrix)

#     def compute_mi_cfm(self, data):
#         """Compute Mutual Information (MI) CFM using mutual_info_regression."""
#         n_channels = data.shape[0]
#         mi_matrix = np.zeros((n_channels, n_channels))
#         for i in range(n_channels):
#             for j in range(i + 1, n_channels):
#                 # Calculate MI for each pair of channels
#                 mi_value = mutual_info_regression(data[i, :].reshape(-1, 1), data[j, :])
#                 mi_matrix[i, j] = mi_matrix[j, i] = mi_value
#         return mi_matrix

#     def compute_te_cfm(self, data):
#         """Compute Transfer Entropy (TE) CFM using pyinform."""
#         n_channels = data.shape[0]
#         te_matrix = np.zeros((n_channels, n_channels))
#         for i in range(n_channels):
#             for j in range(n_channels):
#                 if i != j:
#                     te_matrix[i, j] = pyinform.transferentropy.transfer_entropy(data[i, :], data[j, :], k=1)
#         return te_matrix

#     def fuse_cfms(self, cfm1, cfm2):
#         """Fuse two CFMs by combining upper and lower triangles."""
#         fused_cfm = np.zeros_like(cfm1)
#         n_channels = cfm1.shape[0]
#         for i in range(n_channels):
#             for j in range(n_channels):
#                 if i < j:
#                     fused_cfm[i, j] = cfm1[i, j]  # Upper triangle from cfm1
#                 elif i > j:
#                     fused_cfm[i, j] = cfm2[i, j]  # Lower triangle from cfm2
#         return fused_cfm 
      
# def get_loaders(args):
    
#     if args.task == "subject_identification":
#         splits = ["train", "val_trial"]
#     elif args.task == "emotion_recognition":
#         splits = ["train", "val_trial", "val_subject"]
#     else:
#         raise ValueError(f"Task {args.task} not recognized")
    
#     # Define transforms
#     train_transforms = T.Compose([
#         RandomCrop(args.crop_size),
#         ToTensor(label_interface="long"),
#         Standardize()
#     ])
    
#     test_transforms = T.Compose([
#         ToTensor(label_interface="long"),
#         Standardize()
#     ])


#     # Select dataset
#     subdir = args.data_type
#     if args.data_type == "raw":
#         ext = "fif"
#     elif args.data_type == "pruned":
#         ext = "fif"
#     else:
#         ext = "npy"

#     datasets = {
#         split: EremusDataset(
#             subdir=subdir,
#             split_dir=args.split_dir,
#             split=split,
#             ext = ext,
#             task = args.task,
#             transform=train_transforms if split == "train" else test_transforms
#         )
#         for split in splits
#     }
    
    
#     loaders = {
#         split: DataLoader(
#             dataset,
#             batch_size=args.batch_size if split == "train" else 1,
#             shuffle=True if split == "train" else False,
#             num_workers=args.num_workers,
            
#         )
#         for split, dataset in datasets.items()
#     }

#     return loaders, args

# def get_test_loader(args):
    
#     if args.task == "subject_identification":
#         splits = ["test_trial"]
#     elif args.task == "emotion_recognition":
#         splits = ["test_trial", "test_subject"]
#     else:
#         raise ValueError(f"Task {args.task} not recognized")
    
#     # Define transforms
#     test_transforms = T.Compose([
#         ToTensor(label_interface="long"),
#         Standardize()
#     ])

#     # Select dataset
#     subdir = args.data_type
#     if args.data_type == "raw":
#         ext = "fif"
#     elif args.data_type == "pruned":
#         ext = "fif"
#     else:
#         ext = "npy"

#     datasets = {
#         split: EremusDataset(
#         subdir=subdir,
#         split_dir=args.split_dir,
#         split=split,
#         ext = ext,
#         task = args.task,
#         transform=test_transforms
#         ) for split in splits
#     }
    
#     datasets_no_transform = {
#         split: EremusDataset(
#         subdir=subdir,
#         split_dir=args.split_dir,
#         split=split,
#         ext = ext,
#         task = args.task,
#         transform=None
#         ) for split in splits
#     }
    
#     loaders = {
#         split: DataLoader(
#             dataset,
#             batch_size=1,
#             shuffle=False,
#             num_workers=args.num_workers,
            
#         )
#         for split, dataset in datasets.items()
#     }

#     return datasets_no_transform, loaders, args



import os
import mne
import json
import numpy as np
from tqdm import tqdm
import src.config as config
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from src.eeg_transforms import RandomCrop, ToTensor, Standardize
# from src.utils import EEGToMelSpectrogram
import torchaudio
import torch
import torch.nn.functional as F


mne.set_log_level("ERROR")
 
class EremusDataset(Dataset):
    def __init__(self, subdir, split_dir, split="train", task="subject_identification", ext="fif", transform=None, prefix="",image_transform=False):
        
        self.dataset_dir = config.get_attribute("dataset_path", prefix=prefix)
        self.subdir = os.path.join(subdir, split) if "test" in split else os.path.join(subdir, "train")
        self.split_dir = split_dir
        self.transform = transform
        self.image_transform = image_transform
        self.split = split
        self.label_name = "subject_id" if task == "subject_identification" else "label"
        self.ext = ext
        
        splits = json.load(open(os.path.join(split_dir, f"splits_{task}.json")))
        self.samples = splits[split]
        
        files = []
        for sample in self.samples:
            #path = os.path.join(self.dataset_dir, self.subdir, sample['filename_preprocessed'])
            path = os.path.join(self.dataset_dir, self.subdir, f"{sample['id']}_eeg.{self.ext}")
            files.append(path)
        files = list(set(files))
        #self.files = {f: np.load(f)['arr_0'] for f in files}
        if self.ext == "npy":
            self.files = {f: np.load(f) for f in tqdm(files)}
        elif self.ext == "fif":
            self.files = {f: mne.io.read_raw_fif(f, preload=True).get_data() for f in tqdm(files)}
        else:
            raise ValueError(f"Extension {ext} not recognized")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = self.files[os.path.join(self.dataset_dir, self.subdir, f"{sample['id']}_eeg.{self.ext}")]

        sample = {
            "id": sample['id'],
            "eeg": data,
            # "mel_spec":mel_spectrograms,
            "label": sample[self.label_name] if "test" not in self.split else -1,
        }
        if self.transform:
            sample = self.transform(sample)

        return sample    
      
def get_loaders(args):
    
    if args.task == "subject_identification":
        splits = ["train", "val_trial"]
    elif args.task == "emotion_recognition":
        splits = ["train", "val_trial", "val_subject"]
    else:
        raise ValueError(f"Task {args.task} not recognized")
    
    # Define transforms
    train_transforms = T.Compose([
        # RandomCrop(args.crop_size),
        ToTensor(label_interface="long"),
        # Standardize()
    ])
    
    test_transforms = T.Compose([
        ToTensor(label_interface="long"),
        # Standardize()
    ])


    # Select dataset
    subdir = args.data_type
    if args.data_type == "raw":
        ext = "fif"
    elif args.data_type == "pruned":
        ext = "fif"
    else:
        ext = "npy"

    datasets = {
        split: EremusDataset(
            subdir=subdir,
            split_dir=args.split_dir,
            split=split,
            ext = ext,
            task = args.task,
            transform=train_transforms if split == "train" else test_transforms
        )
        for split in splits
    }
    
    
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size if split == "train" else 1,
            shuffle=True if split == "train" else False,
            num_workers=args.num_workers
        )
        for split, dataset in datasets.items()
    }

    return loaders, args

def get_test_loader(args):
    
    if args.task == "subject_identification":
        splits = ["test_trial"]
    elif args.task == "emotion_recognition":
        splits = ["test_trial", "test_subject"]
    else:
        raise ValueError(f"Task {args.task} not recognized")
    
    # Define transforms
    test_transforms = T.Compose([
        ToTensor(label_interface="long"),
        # Standardize()
    ])

    # Select dataset
    subdir = args.data_type
    if args.data_type == "raw":
        ext = "fif"
    elif args.data_type == "pruned":
        ext = "fif"
    else:
        ext = "npy"

    datasets = {
        split: EremusDataset(
        subdir=subdir,
        split_dir=args.split_dir,
        split=split,
        ext = ext,
        task = args.task,
        transform=test_transforms
        ) for split in splits
    }
    
    datasets_no_transform = {
        split: EremusDataset(
        subdir=subdir,
        split_dir=args.split_dir,
        split=split,
        ext = ext,
        task = args.task,
        transform=None
        ) for split in splits
    }
    
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers
        )
        for split, dataset in datasets.items()
    }

    return datasets_no_transform, loaders, args