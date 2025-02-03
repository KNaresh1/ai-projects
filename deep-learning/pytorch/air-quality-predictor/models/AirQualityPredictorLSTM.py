import torch
from torch import nn


class AirQualityPredictorLSTM(nn.Module):
    def __init__(self, in_features, hidden_size, num_stacked_layers, out_features, dropout_rate=0.15):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        # LSTM layer: The core of the model that learns temporal dependencies in the input sequence
        self.lstm = nn.LSTM(
            in_features,
            hidden_size,
            num_stacked_layers,
            batch_first=True,
            dropout=dropout_rate if num_stacked_layers > 1 else 0  # Dropout between LSTM layers - prevents overfitting
        )

        # Add batch normalization - Helps with training stability, Improves gradient flow
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # Dropout before FC layer
        self.dropout = nn.Dropout(dropout_rate)

        # Fully Connected (FC) layer that maps the output from the LSTM to the final prediction
        self.fc = nn.Linear(hidden_size, out_features)


    '''
        Process the input sequences and propagate them through the LSTM layer, followed by the FC layer
        Handle hidden and cell states (h0 and c0), by either passing the values in or initialize them as zeros
        
        Better device handling in forward pass:
            - Uses input tensor's device for hidden states
            - More robust when using GPU/CPU
    '''
    def forward(self, x, h0=None, c0=None):
        batch_size = x.size(0)

        try:
            self.lstm.flatten_parameters()
        except (RuntimeError, AttributeError):
            pass

        # If hidden and cell states are not provided, initialize them as zeros
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, device=x.device)

        # Forward pass through LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Process the output
        out = out[:, -1, :]  # Get last time step
        out = self.batch_norm(out)  # Apply batch normalization
        out = self.dropout(out)  # Apply dropout
        out = self.fc(out)  # Final linear layer

        return out, hn, cn
