import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        # First 3D Convolution layer
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        # First 3D Max-Pooling and BatchNorm layer
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn1 = nn.BatchNorm3d(32)
        
        # Second 3D Convolution layer
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        # Third 3D Convolution layer
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        # Second 3D Max-Pooling and BatchNorm layer
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.bn2 = nn.BatchNorm3d(64)
        
        # Fully connected layers
        self.emo = nn.Sequential(
            nn.Linear(65536,128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 4)

        )

        self.spk = nn.Sequential(
            nn.Linear(65536, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 26)

        )
        
        
    
    def forward(self, x):
        x = x.unsqueeze(dim=1)
        # First conv, pool, and batch norm layer
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)
        
        # Second conv layer
        x = F.relu(self.conv2(x))
        
        # Third conv, second pool, and batch norm layer
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = self.bn2(x)
        # print(f'[BEFORE FLATTENING]{x.shape}')
        # Flatten and dense layers
        x = x.view(x.size(0), -1)  # Flatten
        emo_logits = self.emo(x)
        spk_logits = self.spk(x)
        
        return emo_logits, spk_logits
