import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(
        self,
        time_window=None,
        eeg_input_dimension=32,
        layers=3,
        kernel_size=3,
        spatial_filters_initial = 256,
        spatial_filters=[256,128,64],
        dilation_filters=[128,64,32],
        activation="relu",
    ):
        super(Model, self).__init__()
        
        self.time_window = time_window
        self.eeg_input_dimension = eeg_input_dimension
        self.layers = layers
        self.kernel_size = kernel_size
        self.spatial_filters = spatial_filters
        self.dilation_filters = dilation_filters
        self.activation = activation

        # Define spatial convolution for EEG input
        self.eeg_proj_1 = nn.Conv1d(eeg_input_dimension, spatial_filters_initial, kernel_size=1)

        # Define dilation layers for EEG and envelope inputs
        self.dilation_layers_eeg = nn.ModuleList()
        
        
        for layer_index in range(layers):
            dilation_rate = kernel_size ** layer_index
            self.dilation_layers_eeg.append(
                nn.Conv1d(spatial_filters[layer_index], dilation_filters[layer_index], kernel_size=kernel_size, dilation=dilation_rate)
            )
            

        # Cosine similarity layers (dot products)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(40128,256),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(256,4)
        )  # Final classification layer with sigmoid activation

    def forward(self, eeg):
        # Spatial convolution on EEG
        # eeg = self.eeg_proj_1(eeg.transpose(1, 2))  # Transpose for Conv1D
        eeg = self.eeg_proj_1(eeg)
        eeg = self.apply_activation(eeg)
        # Dilation layers for EEG and envelope data
        for i in range(self.layers):
            eeg = self.apply_activation(self.dilation_layers_eeg[i](eeg))

        eeg = self.flatten(eeg)
        # print(f'[FLATTEN]{eeg.shape}')
        out = self.fc(eeg)

        return out

    def apply_activation(self, x):
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "tanh":
            return torch.tanh(x)
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

