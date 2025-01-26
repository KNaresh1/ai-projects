import torch.nn as nn
import torch.nn.functional as F

# Model Class that inherits nn.Module

class IrisClassifierNN(nn.Module):

    # Input Layer: 4 features of the flower -->
    # Hidden Layer (H1): n - number of neurons -->
    # Hidden Layer (H2): n - number of neurons -->
    # Output Layer: 3 classes of iris flowers
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__() # Instantiates nn.Module
        self.fc1 = nn.Linear(in_features, h1)   # Layer 1 - Linear Model, fc: fully connected
        self.fc2 = nn.Linear(h1, h2)            # Layer 2
        self.out = nn.Linear(h2, out_features)  # Output Layer


    def forward(self, x):
        x = F.relu(self.fc1(x)) # relu for activation function
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x