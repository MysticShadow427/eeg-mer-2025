import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

from src.models.eegconformer import EEGConformer

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size = (32,32), patch_size=(4,4),dim=128, depth=6, heads=8, mlp_dim=512, channels = 1, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, 4)

    def forward(self, img):
        # print(f'[INPUT]{img.shape}')
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        return x



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        # Define convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        # self.vit = ViT()
        
        # Fully connected layers with batch normalization and dropout
        self.flatten = nn.Flatten()
        self.emo = nn.Sequential(
            # nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=128,nhead=8,dim_feedforward=512,batch_first=True),num_layers=6),
            # nn.Flatten(),
            nn.Linear(8192, 1028),
            nn.ReLU(),
            nn.BatchNorm1d(1028),
            # nn.Dropout(0.25),
            nn.Linear(1028, 128)
        ) 
        self.emo_head = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),
            nn.Linear(128, 4)
        )
        # give out features
        self.spk = nn.Sequential(
            # nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=128,nhead=8,dim_feedforward=512,batch_first=True),num_layers=6),
            # nn.Flatten(),
            nn.Linear(8192, 1028),
            nn.ReLU(),
            nn.BatchNorm1d(1028),
            # nn.Dropout(0.25),
            nn.Linear(1028, 128)
        )
        self.sub_head = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),
            nn.Linear(128, 26)
        )
        
    def forward(self, x):
        # Expand dimensions for grayscale input
        x = x.unsqueeze(dim=1)
        # x = self.vit(x)
        # Convolutional layers with ReLU and batch normalization
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten the output from convolutional layers
        x = self.flatten(x)
        # x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1, 128)
        # print(f'[AFTER FLATTENING]{x.shape}')
        emo_feats = self.emo(x)
        spk_feats = self.spk(x)
        emo_logits = self.emo_head(emo_feats)
        spk_logits = self.sub_head(spk_feats)
        
        return emo_logits, spk_logits, emo_feats, spk_feats
