import torch.nn as nn
import torch.nn.functional as F
from torch.sparse import log_softmax

"""
    Convolutional Neural Network for MNIST Classification
    Image --> Conv Layer 1 --> Pool Layer 1 --> Conv Layer 2 --> Pool Layer 2 --> Fully Connected Layer --> Output Layer
"""
class MNISTClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()

        """
            in_channel: 1 image as input
            out_channel: we ask for 6 filters or feature maps
            kernel_zie: 3 X 3 (Convolutional Filter size)
            stride: 1 - Step it or Stride the filter over the image matrax 1 at a time
        """
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1)
        """
            in_channel: 6 outputs of previous convolutional layer becomes input
            out_channel: 16 (a random number)
        """
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    """
        - Each MNIST image is of 28 X 28 pixels size (2d Image) -  so it is (1, 28, 28)
        - X: 1 4d image - (1, 1, 28, 28): (1 batch, 1 image, 28 X 28)
        - After 1st Convolution, the image size is: (1, 6, 26, 26)
        - As we don't set padding in nn.Conv2d(...), the outer side of pixels got dropped 
          resulting in 26 X 26 instead of 28 X 28 pixels
        - After 1st pooling layer image size is: (1, 6, 13, 13) - 26 / 2 = 13
        - After 2nd Convolution, the image size is: (1, 6, 11, 11)
        - After 2nd pooling layer image size is: (1, 16, 5, 5) - 11 / 2 = 5.5 (rounds down to 5)
        - Flattens to (16 * 5 * 5) and sends to Fully Connection Layers
        - So in brief: ((28 -2) / 2) - 2 / 2 = 5.5
    """
    def forward(self, X):
        # X: 1 4d image - (1, 1, 28, 28): (1 batch, 1 image, 28 X 28)

        # 1st Convolution
        X = F.relu(self.conv1(X))
        # Now Image becomes : (1, 6, 26, 26) - No padding is set, so we loose 2 pixels around the outside of the image
        X = F.max_pool2d(X, 2, 2) # 2 X 2 Kernel
        # Now Image becomes : (1, 6, 13, 13) - 26 /2 = 13

        # 2nd Convolution
        X = F.relu(self.conv2(X))
        # Now Image becomes : (1, 6, 11, 11) - No padding is set, so we loose 2 pixels
        X = F.max_pool2d(X, 2, 2)
        # Now Image becomes : (1, 16, 5, 5) - 11 / 2 = 5.5, since we can't round up the data as we have already deleted
        # the image outside as part of previous steps so it round downs to 5

        # Re-view the data to flatten it out
        X = X.view(-1, 16 * 5 * 5) # -1 is to vary the batch size

        # Fully Connected Layers
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim=1)
