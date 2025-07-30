# model/bertweet_mlp.py

import torch
import torch.nn as nn

class BERTweetMLP(nn.Module):
    """
    Model 2: Uses frozen BERTweet embeddings as input, followed by a 2-layer MLP.
    Input size = 768 (BERTweet hidden size)
    """
    def __init__(self, input_dim=768, hidden_dims=(512, 256), dropout=0.1):
        super(BERTweetMLP, self).__init__()
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
