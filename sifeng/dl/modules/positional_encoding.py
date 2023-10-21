import torch
from torch import nn
from torch.nn.parameter import Parameter

__all__ = [
    "AbsoluteSinusoidalPE",
    "AbsoluteLearnablePE",
]

class AbsoluteSinusoidalPE(nn.Module):
    """The Absolute Sinusoidal Positional Encoding Module
    A. Vaswani et al., “Attention Is All You Need.” arXiv, Aug. 01, 2023. Accessed: Sep. 25, 2023.
    [Online]. Available: http://arxiv.org/abs/1706.03762

    Parameters
    ----------
    embed_dim: int
        the dimension of input embeddings
    dropout: float
        the probability of dropout
    max_len: int, dafault `128`
        the maximum length of positional encoding

    Input shape
    -----------
    [bsz * slen * embed_dim]
    """
    def __init__(self,
                 embed_dim: int,
                 dropout: float,
                 max_len: int = 128):
        super(AbsoluteSinusoidalPE, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.max_len = max_len

        self.dropout = nn.Dropout(p=dropout)
        self.Encoding = torch.empty(1, max_len, embed_dim)
        x = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(max(10000, max_len), torch.arange(0, embed_dim, 2, dtype=torch.float32) / embed_dim)
        self.Encoding[:, :, 0::2] = torch.sin(x)
        self.Encoding[:, :, 1::2] = torch.cos(x)

    def forward(self,
                x: torch.Tensor, # [bsz * slen * embed_dim]
                ) -> torch.Tensor:
        bsz, slen, embed_dim = x.shape
        return x + self.dropout(self.Encoding[:, :slen, :])

class AbsoluteLearnablePE(nn.Module):
    """The Absolute Positional Encoding Module
    A. Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at
    Scale.” arXiv, Jun. 03, 2021. Accessed: Sep. 28, 2023. [Online]. Available:
    http://arxiv.org/abs/2010.11929

    Parameters
    ----------
    embed_dim: int
        the dimension of input embeddings
    dropout: float
        the probability of dropout
    max_len: int, dafault `128`
        the maximum length of positional encoding

    Input shape
    -----------
    [bsz * slen * embed_dim]
    """
    def __init__(self,
                 embed_dim: int,
                 dropout: float = 0.0,
                 max_len: int = 128,
                 ) -> None:
        super(AbsoluteLearnablePE, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.max_len = max_len

        self.dropout = nn.Dropout(p=dropout)
        self.Encoding = Parameter(torch.rand(1, max_len, embed_dim))

    def forward(self,
                x: torch.Tensor, # [bsz * slen * embed_dim]
                ) -> torch.Tensor:
        bsz, slen, embed_dim = x.shape
        return x + self.dropout(self.Encoding[:, :slen, :])
