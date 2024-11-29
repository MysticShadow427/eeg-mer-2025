import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import math

# Convolution module
class PatchEmbedding_C(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        # print("Patch Embedding")
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        # print(f"[INPUT] {x.shape}")
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention_C(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # print('Multihead Attention')
        # print(f"[INPUT] {x.shape}")
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        # print(f"[OUTPUT] {out.shape}")
        return out


class ResidualAdd_C(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class DilatedCNN(nn.Sequential):
    def __init__(self):
        super().__init__(
            
            nn.Conv1d(in_channels=32,out_channels=64,dilation=1,kernel_size=1,stride=1),
            nn.GELU(),
            nn.Conv1d(in_channels=64,out_channels=64,dilation=2,kernel_size=1,stride=1),
            nn.GELU(),
            nn.Conv1d(in_channels=64,out_channels=128,dilation=4,kernel_size=1,stride=1),
            nn.GELU(),
            nn.Conv1d(in_channels=128,out_channels=128,dilation=8,kernel_size=1,stride=1),
            nn.GELU()
        
        )

class FeedForwardBlock_C(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU_C(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock_C(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd_C(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention_C(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd_C(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock_C(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder_C(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock_C(emb_size) for _ in range(depth)])

# # Set number of classes
#     if args.task == "emotion_recognition":
#         args.num_classes = 4
#     elif args.task == "subject_identification":
#         args.num_classes = 26
class ClassificationHead_C(nn.Sequential):
    def __init__(self, emb_size, num_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )
        self.bidirectional = True
        self.num_layers = 1
        self.fc = nn.Sequential(
            # nn.Linear(2440, 256),
            
            nn.Linear(34760,256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )
        # add 1d dialted for the emotion recognition task
        # self.projection = nn.Linear(34760,256)
        # self.dilated_cnns = nn.Sequential(
        #     nn.Conv1d(in_channels=40,out_channels=64,dilation=1,kernel_size=1,stride=1),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=64,out_channels=64,dilation=2,kernel_size=1,stride=1),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=64,out_channels=128,dilation=4,kernel_size=1,stride=1),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=128,out_channels=128,dilation=8,kernel_size=1,stride=1),
        #     nn.ReLU()
        # )
        # self.pool = nn.AdaptiveAvgPool1d(1) # reduces tensor to [batch_size, channels, 1].
        # self.c = nn.Sequential(
        #     nn.Linear(128,64),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(64,4)
        # )


        # self.c = nn.Linear(64,4) # 26 for subject_identification task
        

    
    # def forward(self,x):
    #     x = x.permute(0,2,1)
    #     x = self.dilated_cnns(x)
    #     x = x.permute(0,2,1)
    #     x = self.pool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.c(x)
        
        
    #     # return out, x
    #     return x
    def forward(self,x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out
        
        
# seeing page 5 of the eeg contrastive learning paper, use the projection network the lstm one they have used , need to change the trainer too man!
class FinalHead_C(nn.Sequential):
    def __init__(self, emb_size):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, 256)
        )
        self.fc = nn.Sequential(
            # nn.Linear(2440, 256),
            nn.Linear(34760,256) # *20 kar dena input channels when we do this simple classification
            
        )
        

    def forward(self, x):
        # print('Classification Head')
        # print(f"[INPUT] {x.shape}")
        x = x.contiguous().view(x.size(0), -1)
        # print(f"[INPUT] {x.shape}")
        out = self.fc(x)
        # print(f"[LOGITS] {out.shape}")
        # print(f"[FEATS] {x.shape}")
        return out #,x# modify this or the loss function if you are simclr


class Model(nn.Sequential):
    def __init__(self,args,emb_size=40, depth=4, num_classes=2, **kwargs):
        args_defaults=dict(emb_size=40, depth=4, num_classes=2, verbose=False)
        for arg,default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)
        super().__init__(
            DilatedCNN(),
            PatchEmbedding_C(emb_size),
            TransformerEncoder_C(depth, emb_size),
            ClassificationHead_C(emb_size, num_classes)
        )

class EEGConformer(nn.Sequential):
    def __init__(self,args,emb_size=40, depth=4, num_classes=2, **kwargs):
        args_defaults=dict(emb_size=40, depth=4, num_classes=2, verbose=False)
        for arg,default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)
        super().__init__(

            PatchEmbedding_C(emb_size),
            TransformerEncoder_C(depth, emb_size),
            FinalHead_C(emb_size) # 256 dim output
        )
