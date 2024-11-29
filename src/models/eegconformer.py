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
        self.p1 = nn.LSTM(batch_first=True, bidirectional=True,hidden_size=256,num_layers=1,input_size=40)
        self.p2 = nn.LSTM(batch_first=True, bidirectional=True,hidden_size=128,num_layers=1,input_size=40)
        self.p3 = nn.LSTM(batch_first=True, bidirectional=True,hidden_size=64,num_layers=1,input_size=40)

        self.d = nn.Sequential(
            nn.Linear(896,512),
            nn.ReLU(),
            nn.Linear(512,256)
        )

        self.c = nn.Linear(256,26) # 26 for subject_identification task
        

    
    def forward(self,x):
        # print('Classification Head')
        # print(f"[INPUT] {x.shape}")
        x = x.transpose(1, 2)  
        x1 = F.avg_pool1d(x, kernel_size=2, stride=2)
        x2 = F.avg_pool1d(x1, kernel_size=2, stride=2)
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1,2)
        x = x.transpose(1, 2)

        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, 256).to('cuda')
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, 256).to('cuda')
        o1, (h_n1, c_n1) = self.p1(x, (h0, c0))

        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, 128).to('cuda')
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, 128).to('cuda')
        o2, (h_n2, c_n2) = self.p2(x1, (h0, c0))

        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, 64).to('cuda')
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, 64).to('cuda')
        o3, (h_n3, c_n3) = self.p3(x2, (h0, c0))

        fh_1 = h_n1[0, :, :]  # Forward direction hidden state
        bh_1 = h_n1[1, :, :] # Backward direction hidden state

        fh_2 = h_n2[0, :, :]  # Forward direction hidden state
        bh_2 = h_n2[1, :, :] # Backward direction hidden state

        fh_3 = h_n3[0, :, :]  # Forward direction hidden state
        bh_3 = h_n3[1, :, :] # Backward direction hidden state

        f1 = torch.cat((fh_1, bh_1), dim=1)  # [batch_size, 2*lstm_hidden_dim1]
        f2 = torch.cat((fh_2, bh_2), dim=1)  # [batch_size, 2*lstm_hidden_dim2]
        f3 = torch.cat((fh_3, bh_3), dim=1)

        ch = torch.cat((f1, f2, f3), dim=1)  
        # Shape: [batch_size, 2 * (lstm_hidden_dim1 + lstm_hidden_dim2 + lstm_hidden_dim3)]
        dh = self.d(ch)
        logits = self.c(dh)

        

        # x = x.contiguous().view(x.size(0), -1)
        # out = self.fc(x)
        # x = self.projection(x)
        
        # return out, x
        return logits, dh
        
        
class FinalHead_C(nn.Sequential):
    def __init__(self, emb_size=40):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, 256)
        )
        self.fc = nn.Sequential(
            # nn.Linear(2440, 256),
            nn.Linear(34760,256), # *20 kar dena input channels when we do this simple classification
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,4)
        )
        

    def forward(self, x):
        # print('Classification Head')
        # print(f"[INPUT] {x.shape}")
        x = x.contiguous().view(x.size(0), -1)
        # print(f"[INPUT] {x.shape}")
        out = self.fc(x)
        # print(f"[LOGITS] {out.shape}")
        # print(f"[FEATS] {x.shape}")
        return out #,x# 

class Head(nn.Sequential):
    def __init__(self):
        super().__init__()
    
        self.emo = nn.Sequential(
            # nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=128,nhead=8,dim_feedforward=512,batch_first=True),num_layers=6),
            # nn.Flatten(),
            nn.Linear(34760, 1028),
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
            nn.Linear(34760, 1028),
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
        x = x.contiguous().view(x.size(0), -1)
        emo_feats = self.emo(x)
        spk_feats = self.spk(x)
        emo_logits = self.emo_head(emo_feats)
        spk_logits = self.sub_head(spk_feats)

        
        return emo_logits, spk_logits, emo_feats, spk_feats

class Model(nn.Sequential):
    def __init__(self,args,emb_size=40, depth=4, num_classes=2, **kwargs):
        args_defaults=dict(emb_size=40, depth=4, num_classes=2, verbose=False)
        for arg,default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)
        super().__init__(

            PatchEmbedding_C(emb_size),
            TransformerEncoder_C(depth, emb_size),
            FinalHead_C(),
            # Head()
            # ClassificationHead_C(emb_size, num_classes) use this for emotion recogntion

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
