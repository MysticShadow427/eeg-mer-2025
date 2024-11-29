import os
import mne
import json
import numpy as np
from tqdm import tqdm
import src.config as config
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from src.eeg_transforms import RandomCrop, ToTensor, Standardize
from src.utils import EEGToMelSpectrogram
import torchaudio
import torch
import torch.nn.functional as F


mne.set_log_level("ERROR")
 
class EremusDataset(Dataset):
    def __init__(self, subdir, split_dir, split="train", task="subject_identification", ext="fif", transform=None, prefix="",image_transform=True):
        
        self.dataset_dir = config.get_attribute("dataset_path", prefix=prefix)
        self.subdir = os.path.join(subdir, split) if "test" in split else os.path.join(subdir, "train")
        self.split_dir = split_dir
        self.transform = transform
        self.image_transform = image_transform
        self.mel_transform = EEGToMelSpectrogram()
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=80)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=80)
        self.to_tensor = T.ToTensor()
        self.standardize = Standardize()
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
        # eeg_data = data.unsqueeze(0)
        # mel_spectrograms = self.mel_transform.eeg_to_mel(eeg_data)
        # mel_spectrograms = F.interpolate(mel_spectrograms, size=(128,80), mode='bilinear', align_corners=False)
        # mel_spectrograms = mel_spectrograms.squeeze(0)
        sample = {
            "id": sample['id'],
            "eeg": data,
            # "mel_spec":mel_spectrograms,
            "label": sample[self.label_name] if "test" not in self.split else -1,
        }
        if self.transform:
            sample = self.transform(sample)
        if self.image_transform:
            # eeg_data = sample['eeg'] 
            # eeg_data = self.to_tensor(eeg_data)
            # eeg_data = self.standardize(eeg_data)
            # print(f'[EEG] {eeg_data.shape}')
            eeg_data = torch.tensor(data,dtype = torch.float).unsqueeze(0)
            # print(f'[EEG after unsqueeze] {eeg_data.shape}')
            mel_spectrograms = self.mel_transform.eeg_to_mel(eeg_data)
            # print(f'[EEG Mel Spec] {mel_spectrograms.shape}')
            # mel_spectrograms = F.interpolate(mel_spectrograms, size=(128,80), mode='bilinear', align_corners=False)
            # print(f'[EEG Mel Spec resized] {mel_spectrograms.shape}')
            mel_spectrograms = mel_spectrograms.squeeze(0)
            # print(f'[EEG Mel Spec final] {mel_spectrograms.shape}')
            # sample['mel_spec'] = mel_spectrograms
            # Resize the time dimension to 1024 while keeping num_mels (128) the same
            
            if self.split=='train':
                mel_spectrograms = self.time_masking(mel_spectrograms)
                mel_spectrograms = self.freq_masking(mel_spectrograms)
                sample['mel_spec'] = mel_spectrograms
            else:
                sample['mel_spec'] = mel_spectrograms

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
        RandomCrop(args.crop_size),
        ToTensor(label_interface="long"),
        Standardize()
    ])
    
    test_transforms = T.Compose([
        ToTensor(label_interface="long"),
        Standardize()
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
            num_workers=args.num_workers,
            collate_fn=collate_fn
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
        Standardize()
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
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
        for split, dataset in datasets.items()
    }

    return datasets_no_transform, loaders, args

from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    print(f'[BATCH]{batch}')
    print()

    features = zip(*batch)
    print(f'[FEATURES]{features}')
    i = [item['id'] for item in features]
    d = [item['eeg'] for item in features]
    mels = [item['mel_spec'] for item in features]
    l = [item['label'] for item in features]
    mels = pad_sequence(mels, batch_first=True)
    d = torch.stack(d)
    i = torch.stack(i)
    l = torch.stack(l)

    return {
        'id':i,
        'eeg':d,
        'mel_spec':mels,
        'label':l
    }