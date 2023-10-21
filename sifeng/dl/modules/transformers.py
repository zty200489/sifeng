import torch
from torch import nn

__all__ = [
    "FeedForwardLayer",
]

class FeedForwardLayer(nn.Module):
    """The Feed-Forward Layer

    Parameters
    ----------
    embed_dim: int
        the dimension of input embeddings
    activation: torch.nn.Module, default torch.nn.ReLU
        the activation module used in the feed-forward sandwich
    elementwise_affine: bool, default `False`
        whether to enable element-wise affine in layer norm
    *args, **kwargs:
        Parameters for the activation function

    Input shape
    -----------
    [bsz * ?? * embed_dim]
    """
    def __init__(self,
                 embed_dim: int,
                 activation: nn.Module = nn.ReLU,
                 *args, **kwargs) -> None:
        super(FeedForwardLayer, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            activation(*args, **kwargs),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self,
                x: torch.Tensor, # [bsz * ?? * embed_dim]
                ) -> torch.Tensor:
        return self.sequential(x)
