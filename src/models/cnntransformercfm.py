import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128,nhead=8,batch_first=True,dim_feedforward=512)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,num_layers=4)
        # Fully connected layers with batch normalization and dropout
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8192, 1028) #8192 (= 8 × 8 × 128)
        self.bn_fc1 = nn.BatchNorm1d(1028)
        self.dropout1 = nn.Dropout(0.25)
        
        self.fc2 = nn.Linear(1028, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.25)
        
        self.fc3 = nn.Linear(128, 4)  # Output layer for emotion recognition
        
    def forward(self, x):
        # Expand dimensions for grayscale input
        x = x.unsqueeze(dim=1)
        
        # Convolutional layers with ReLU and batch normalization
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

    
        # Reshape for transformer encoder: [batch_size, 64, 128]
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1, 128)  # [batch_size, 64, 128]
        # Reshape [bs,n_c,h,w] for transformer
        # x = x.reshape(x.size(0),x.size(1),-1) # [bs,n_c,h*w]
        x = self.encoder(x)

        # Flatten the output from transformer encoder layers
        x = self.flatten(x)
        
        # Fully connected layers with ReLU, batch normalization, and dropout
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x
