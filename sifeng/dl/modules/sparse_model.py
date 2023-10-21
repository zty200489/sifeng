import torch
from torch import nn
from .transformers import FeedForwardLayer

from typing import Optional, Callable

class MixtureOfExpertsBlock(nn.Module):
    """The Mixture of Experts Module
    - MoE
      N. Shazeer et al., “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts
      Layer.” arXiv, Jan. 23, 2017. Accessed: Sep. 27, 2023. [Online]. Available:
      http://arxiv.org/abs/1701.06538
    - FastMoE
      J. He, J. Qiu, A. Zeng, Z. Yang, J. Zhai, and J. Tang, “FastMoE: A Fast Mixture-of-Expert Tra
      ining System.” arXiv, Mar. 24, 2021. Accessed: Sep. 28, 2023. [Online]. Available:
      http://arxiv.org/abs/2103.13262
    - TODO: ST-MoE
    B. Zoph et al., “ST-MoE: Designing Stable and Transferable Sparse Expert Models.” arXiv, Apr.
    29, 2022. Accessed: Sep. 27, 2023. [Online]. Available: http://arxiv.org/abs/2202.08906

    Parameters
    ----------
    embed_dim: int
        the dimension of input embeddings
    num_experts: int
        the number of expert modules
    topk: int, default `2`
        the number of topk experts to activate
    gate: Optional[Callable[[torch.Tensor], torch.Tensor]], default `None`
        the gate layer, if `None` default to simple linear transformation
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
                 gate: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 expert: nn.Module = FeedForwardLayer,
                 *args, **kwargs) -> None:
        super(MixtureOfExpertsBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.topk = topk

        self.softmax = nn.Softmax(dim=-1)
        self.gate = gate if gate is not None else nn.Linear(embed_dim, self.num_experts)
        self.experts = nn.ModuleList([expert(embed_dim=embed_dim, *args, **kwargs) for _ in range(self.num_experts)])

    def forward(self,
                x: torch.Tensor, # [bsz * ?? * embed_dim]
                ) -> torch.Tensor:
        device = x.device
        fsz = x.shape[:-1] # fsz = [bsz * ??]
        x = x.flatten(start_dim=0, end_dim=-2) # [fsz * embed_dim]
        gate = self.gate(x) # [fsz * num_experts]
        topk_values, topk_indices = torch.topk(gate, k=self.topk, dim=-1, sorted=True) # [fsz * topk], [fsz * topk]
        topk_weight = self.softmax(topk_values) # [fsz * topk]
        mask = torch.zeros([topk_weight.shape[0], self.num_experts], device=device, dtype=topk_weight.dtype).scatter(dim=1, index=topk_indices, src=topk_weight) # [fsz * num_experts]
        activators = [mask[:, i].nonzero().squeeze(-1) for i in range(self.num_experts)] # List([?], len=num_experts)
        weights = [mask[:, i].index_select(dim=0, index=activators[i]) for i in range(self.num_experts)]
        inputs = [x.index_select(dim=0, index=activators[i]) for i in range(self.num_experts)] # List([? * embed_dim], len=num_experts)
        outputs = [torch.mul(self.experts[i](inputs[i]), weights[i].unsqueeze(dim=-1)) for i in range(self.num_experts)] # List([? * embed_dim], len=num_experts)
        x = torch.zeros(x.shape[0], outputs[0].shape[-1], device=device, dtype=x.dtype) # [fsz * embed_dim]
        for i in range(self.num_experts):
            x = x.scatter_add(0, activators[i].unsqueeze(dim=-1).repeat_interleave(outputs[0].shape[-1], dim=-1), outputs[i]) # [fsz * embed_dim]
        x = x.unflatten(dim=0, sizes=fsz) # [bsz * ?? * embed_dim]
        return x
