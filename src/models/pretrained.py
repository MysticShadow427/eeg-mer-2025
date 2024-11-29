import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        # Load the EfficientNet model without pre-trained weights
        self.feature_extractor = models.convnext_tiny()

        # Modify the first convolutional layer to accept 32 input channels
        # self.feature_extractor.features[0][0] = nn.Conv2d(
        #     in_channels=32, 
        #     out_channels=self.feature_extractor.features[0][0].out_channels, 
        #     kernel_size=self.feature_extractor.features[0][0].kernel_size, 
        #     stride=self.feature_extractor.features[0][0].stride, 
        #     padding=self.feature_extractor.features[0][0].padding, 
        #     bias=self.feature_extractor.features[0][0].bias
        # )
        first_conv_layer = self.feature_extractor.features[0][0]
        self.feature_extractor.features[0][0] = nn.Conv2d(
            in_channels=32,
            out_channels=first_conv_layer.out_channels,
            kernel_size=first_conv_layer.kernel_size,
            stride=first_conv_layer.stride,
            padding=first_conv_layer.padding,
            bias=True if first_conv_layer.bias is not None else False  # Correctly handle the bias as a boolean
        )
        
        # Remove the top fully connected layers
        self.feature_extractor = nn.Sequential(
            *list(self.feature_extractor.children())[:-1]
        )

        self.feature_extractor = self.feature_extractor[0]
        
        # Flatten layer
        # self.flatten = nn.Flatten()
        
        # # Fully connected neural network
        # self.fc = nn.Sequential(
        #     nn.Linear(1280, 512),  # Adjust input size if different backbone is used
        #     nn.ReLU(),
        #     nn.Dropout(p=0.4),
        #     nn.Linear(512, 4)
        # )
        self.num_layers = 1
        self.bidirectional = True

        self.p1 = nn.LSTM(batch_first=True, bidirectional=True,hidden_size=256,num_layers=1,input_size=768*4)
        self.p2 = nn.LSTM(batch_first=True, bidirectional=True,hidden_size=128,num_layers=1,input_size=768*4)
        self.p3 = nn.LSTM(batch_first=True, bidirectional=True,hidden_size=64,num_layers=1,input_size=768*4)

        self.d = nn.Sequential(
            nn.Linear(896,128),
            nn.ReLU(),
            nn.Linear(128,32)
        )

        self.c = nn.Linear(32,4)
    
    # def forward(self, x):
    #     x = self.feature_extractor(x)
    #     x = self.flatten(x)
    #     x = self.fc(x)
    #     return x

    def forward(self,x):
        x = self.feature_extractor(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
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

        return logits

