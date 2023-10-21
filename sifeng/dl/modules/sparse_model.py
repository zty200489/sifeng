import torch, numpy as np
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from .transformers import FeedForwardLayer

class MixtureOfExpertsBlock(nn.Module):
    """The Mixture of Experts Module

    Parameters
    ----------
    embed_dim: int
        the dimension of input embeddings
    num_experts: int
        the number of expert modules
    topk: int, default `2`
        the number of topk experts to activate
    expert: torch.nn.Module, default `sifeng.dl.FeedForwardLayer`
        the choice of expert module
    *args, **kwargs:
        parameters for your choice of the expert module

    Input shape
    -----------
    [bsz * ?? * embed_dim]

    """
    def __init__(self,
                 embed_dim: int,
                 num_experts: int,
                 topk: int = 2,
                 expert: nn.Module = FeedForwardLayer,
                 *args, **kwargs) -> None:
        super(MixtureOfExpertsBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.topk = topk

        self.softmax = nn.Softmax(dim=-1)
        self.W_gate = Parameter(torch.empty(embed_dim, self.num_experts))
        self.experts = nn.ModuleList([expert(embed_dim=embed_dim, *args, **kwargs) for _ in range(self.num_experts)])

        self._init_params()

    def _init_params(self):
        xavier_uniform_(self.W_gate)

    def forward(self,
                x: torch.Tensor, # [bsz * ?? * embed_dim]
                ) -> torch.Tensor:
        gate = torch.matmul(x, self.W_gate) # [bsz * ?? * num_experts]
        kth = torch.min(torch.topk(gate, k=self.topk, dim=-1, sorted=False).values, dim=-1, keepdim=True).values # [bsz * ?? * 1]
        mask = self.softmax(torch.where(torch.ge(gate, kth), gate, -np.inf)) # [bsz * ?? * num_experts]
        x = x.unsqueeze(dim=-2).repeat(1, 1, self.num_experts, 1) # [bsz * ?? * num_experts * embed_dim]
        x = torch.concat([self.experts[i](x.select(dim=-2, index=i).unsqueeze(dim=-2)) for i in range(self.num_experts)], dim=-2) # [bsz * ?? * num_experts * embed_dim]
        x = torch.mul(x, mask.unsqueeze(dim=-1)) # [bsz * ?? * num_experts * embed_dim]
        return x.mean(dim=-2) # [bsz * ?? * embed_dim]
