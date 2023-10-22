"""
CTTS

Z. Zeng, R. Kaur, S. Siddagangappa, S. Rahimi, T. Balch, and M. Veloso, “Financial Time Series Forecasting using CNN and Transformer.” arXiv, Apr. 10, 2023. Accessed: Sep. 24, 2023. [Online]. Available: http://arxiv.org/abs/2304.04912
"""

import torch
from torch import nn
from ..modules import MultiheadedSelfAttentionModule

__all__ = ["CTTS"]

class CTTS(nn.Module):
    def __init__(self,
                 window: int = 80,
                 embed_dim: int = 128,
                 kernel_size: int = 16,
                 stride: int = 8,
                 attn_depth: int = 4,
                 attn_heads: int = 4,
                 dropout: float = 0.3,
                 categories: int = 2,
                 ) -> None:
        super(CTTS, self).__init__()
        self.window = window
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.attn_depth = attn_depth
        self.attn_heads = attn_heads
        self.dropout = dropout
        self.categories = categories
        self.token_len = (window - kernel_size) // stride + 1
        # Define modules
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=kernel_size, stride=stride)
        self.attns = nn.Sequential(*(MultiheadedSelfAttentionModule(num_head=attn_heads, embed_dim=embed_dim, dropout=dropout) for _ in range(attn_depth)))
        self.mlp = nn.Sequential(
            nn.Linear(self.token_len * embed_dim, self.token_len * embed_dim * 2 // 3),
            nn.GELU(),
            nn.Linear(self.token_len * embed_dim * 2 // 3, categories),
        )

    def forward(self,
                x, # [bsz, window]
                ) -> torch.Tensor:
        assert x.shape[1] == self.window, "Wrong input tensor shape."
        # Nornmalize and reshape
        x = (x - x.min(dim=1, keepdim=True).values) /                                               \
            (x.max(dim=1, keepdim=True).values - x.min(dim=1, keepdim=True).values)                 # [bsz, window]
        x = x.unsqueeze(dim=1)                                                                      # [bsz, 1, window]
        # Extract latent dependencies
        x = self.conv1d(x)                                                                          # [bsz, embed_dim, token_len]
        x = self.attns(x.transpose(1, 2))                                                           # [bsz, token_len, embed_dim]
        # Predict
        x = self.mlp(x.flatten(1, 2))                                                               # [bsz, categories]
        return x
