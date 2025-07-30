# model/stacked_model.py

import torch
import torch.nn as nn

class StackedModel(nn.Module):
    """
    Model 4: Concatenates BERTweet embeddings (768-d) and projected user-level features (128-d)
    Input: [batch_size, 896]
    Structure: 2-layer MLP -> binary output
    """
    def __init__(self, input_dim=896, hidden_dims=(512, 256), dropout=0.2):
        super(StackedModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)
