from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from src.augmentations import ft_surrogate,gaussian_noise

# https://github.com/HobbitLong/SupContrast/blob/master/losses.py
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR, just dont give labels in the loss function while calculating"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits

def compute_cross_entropy(p, q):
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return - loss.mean()


class MultiPosConLoss(nn.Module):
    """
    Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
    Not Useful for us
    """

    def __init__(self, temperature=0.1):
        super(MultiPosConLoss, self).__init__()
        self.temperature = temperature
        self.logits_mask = None
        self.mask = None
        self.last_local_batch_size = None

    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, feats, labels):
        # feats = outputs['feats']    # feats shape: [B, D]
        # labels = outputs['labels']    # labels shape: [B]

        device = (torch.device('cuda')
                  if feats.is_cuda
                  else torch.device('cpu'))

        feats = F.normalize(feats, dim=-1, p=2)
        local_batch_size = feats.size(0)
        world_size = dist.get_world_size()
        gathered_feats = [torch.zeros_like(feats) for _ in range(world_size)]
        dist.all_gather(gathered_feats, feats)
        all_feats = torch.cat(gathered_feats, dim=0)
        # all_feats = torch.cat(torch.distributed.nn.all_gather(feats), dim=0)
        all_labels = concat_all_gather(labels)  # no gradient gather

        # compute the mask based on labels
        if local_batch_size != self.last_local_batch_size:
            mask = torch.eq(labels.view(-1, 1),
                            all_labels.contiguous().view(1, -1)).float().to(device)
            self.logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(mask.shape[0]).view(-1, 1).to(device) +
                local_batch_size * get_rank(),
                0
            )

            self.last_local_batch_size = local_batch_size
            self.mask = mask * self.logits_mask

        mask = self.mask

        # compute logits
        logits = torch.matmul(feats, all_feats.T) / self.temperature
        logits = logits - (1 - self.logits_mask) * 1e9

        # optional: minus the largest logit to stablize logits
        logits = stablize_logits(logits)

        # compute ground-truth distribution
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        loss = compute_cross_entropy(p, logits)

import torch
import torchaudio
import torchaudio.transforms as T
class EEGToMelSpectrogram:
    def __init__(self, sample_rate=128, n_fft=256, win_length=None, hop_length=16, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
    
    def eeg_to_mel(self, eeg_data):
        batch_size, num_channels, num_samples = eeg_data.shape
        mel_spectrograms = []

        for i in range(num_channels):
            channel_data = eeg_data[:, i, :]  # Extract data for the i-th channel
            mel_spectrogram = self.mel_spectrogram_transform(channel_data)  # Shape: [batch_size, num_mels, time_steps]
            mel_spectrograms.append(mel_spectrogram)

        mel_spectrograms = torch.stack(mel_spectrograms, dim=1)  # Shape: [batch_size, num_channels, num_mels, time_steps]
        return mel_spectrograms

import random
def augment_features(eeg, labels, model,n_views=3,std_range=(0.0, 1.0), phase_noise_magnitude_range=(0.0, 1.0),random_state=42):
    hidden_dim = 256
    bs, n_c, n_t = eeg.shape
    all_views = torch.zeros(bs, n_views, hidden_dim)
    for i in range(n_views):
        std = random.uniform(*std_range)
        augmented_eeg = gaussian_noise(eeg,std=std,random_state=random_state)
        phase_noise_magnitude = random.uniform(*phase_noise_magnitude_range)
        augmented_eeg, _ = ft_surrogate(augmented_eeg,labels,phase_noise_magnitude=phase_noise_magnitude,channel_indep=False,random_state=random_state)
        _, embeddings = model(augmented_eeg)
        all_views[:, i, :] = embeddings


    return all_views
    

    # gaussian noise


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Initialize Focal Loss
        :param alpha: Weighting factor for the class, balances importance of positive/negative examples (default is 1)
        :param gamma: Focusing parameter, reduces the loss for well-classified examples, putting more focus on hard examples
        :param reduction: Specifies the reduction to apply to the output ('none', 'mean', or 'sum')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none',label_smoothing = 0.2)
        pt = torch.exp(-ce_loss)  # probability of correct class

        # Compute focal loss
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

import numpy as np
# from mne.connectivity import spectral_connectivity
from mne_connectivity import SpectralConnectivity
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
import pyinform     

def compute_pcc_cfm(data):
        """Compute Pearson Correlation Coefficient (PCC) CFM."""
        n_channels = data.shape[0]
        pcc_matrix = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                pcc, _ = pearsonr(data[i, :], data[j, :])
                pcc_matrix[i, j] = pcc_matrix[j, i] = pcc
        return pcc_matrix

def compute_plv_cfm(data):
    """Compute Phase Locking Value (PLV) CFM."""
    Freq_Bands = {"theta": [4.0, 8.0], "alpha": [8.0, 13.0], "beta": [13.0, 30.0]}
    n_freq_bands = len(Freq_Bands)
    min_freq = np.min(list(Freq_Bands.values()))
    max_freq = np.max(list(Freq_Bands.values()))

    # Provide the freq points
    freqs = np.linspace(min_freq, max_freq, int((max_freq - min_freq) * 4 + 1))
    data_reshaped = data[np.newaxis, :, :]  # Shape (1, n_channels, n_times)
    plv_matrix, _, _, _, _ = SpectralConnectivity(
        data_reshaped, method='plv',freqs=freqs,
    )
    return np.squeeze(plv_matrix)

def compute_mi_cfm(data):
    """Compute Mutual Information (MI) CFM using mutual_info_regression."""
    n_channels = data.shape[0]
    mi_matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            # Calculate MI for each pair of channels
            mi_value = mutual_info_regression(data[i, :].reshape(-1, 1), data[j, :])
            mi_matrix[i, j] = mi_matrix[j, i] = mi_value
    return mi_matrix

def compute_te_cfm(data):
    """Compute Transfer Entropy (TE) CFM using pyinform."""
    n_channels = data.shape[0]
    te_matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(n_channels):
            if i != j:
                te_matrix[i, j] = pyinform.transferentropy.transfer_entropy(data[i, :], data[j, :], k=1)
    return te_matrix

def fuse_cfms(cfm1, cfm2):
    """Fuse two CFMs by combining upper and lower triangles."""
    fused_cfm = np.zeros_like(cfm1)
    n_channels = cfm1.shape[0]
    for i in range(n_channels):
        for j in range(n_channels):
            if i < j:
                fused_cfm[i, j] = cfm1[i, j]  # Upper triangle from cfm1
            elif i > j:
                fused_cfm[i, j] = cfm2[i, j]  # Lower triangle from cfm2
    return fused_cfm 

# def orthogonality_loss(emotion_features, subject_features):
#     # Compute the cosine similarity between the two feature sets
#     dot_product = torch.sum(emotion_features * subject_features, dim=1)
#     norm_emotion = torch.norm(emotion_features, p=2, dim=1)
#     norm_subject = torch.norm(subject_features, p=2, dim=1)
#     cosine_similarity = dot_product / (norm_emotion * norm_subject + 1e-8)
#     # Minimize cosine similarity, encouraging orthogonality
#     return torch.mean(cosine_similarity ** 2)

class orthogonality_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,emotion_features, subject_features):
        dot_product = torch.sum(emotion_features * subject_features, dim=1)
        norm_emotion = torch.norm(emotion_features, p=2, dim=1)
        norm_subject = torch.norm(subject_features, p=2, dim=1)
        cosine_similarity = dot_product / (norm_emotion * norm_subject + 1e-8)
        # Minimize cosine similarity, encouraging orthogonality
        return torch.mean(cosine_similarity ** 2)

# def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
#     n_samples = int(source.size()[0]) + int(target.size()[0])
#     total = torch.cat([source, target], dim=0)
#     L2_distance = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)
#     if fix_sigma:
#         bandwidth = fix_sigma
#     else:
#         bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
#     bandwidth /= kernel_mul ** (kernel_num // 2)
#     bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
#     kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
#     return sum(kernel_val)

# def mmd_loss(source, target):
#     source_kernel = gaussian_kernel(source, source)
#     target_kernel = gaussian_kernel(target, target)
#     cross_kernel = gaussian_kernel(source, target)
#     return torch.mean(source_kernel + target_kernel - 2 * cross_kernel)

class mmd_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def gaussian_kernel(self,source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        L2_distance = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def forward(self,source, target):
        source_kernel = self.gaussian_kernel(source, source)
        target_kernel = self.gaussian_kernel(target, target)
        cross_kernel = self.gaussian_kernel(source, target)
        return torch.mean(source_kernel + target_kernel - 2 * cross_kernel)


        