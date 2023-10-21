import torch
from torch.nn import Module
import torch.nn.functional as F

from typing import Optional

__all__ = [
    "FocalLoss",
]

class FocalLoss(Module):
    """An implementation of focal loss.
    The focal loss is computated by:

    .. math::
        Loss(x, class) = - \alpha (1-\mathrm{softmax}(x)[class])^{\gamma} \log(\mathrm{softmax}(x)[class])

    T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal Loss for Dense Object
    Detection.” arXiv, Feb. 07, 2018. Accessed: Oct. 17, 2023. [Online]. Available:
    http://arxiv.org/abs/1708.02002

    Parameters
    ----------
    alpha: Optional[torch.Tensor], default `None`:
        Weighting factors in range (0,1) to balance positive vs negative examples. Only one between
    `alpha` and `freq` can be passed.
    freq: Optional[torch.Tensor], default `None`:
        The frequency which each category appears. Only one between `alpha` and `freq` can be passed.
    gamma: float, default `2.0`
        Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
    reduction: str, default `mean`
        - `none`: No reduction will be applied to the output.
        - `mean`: The output will be averaged.
        - `sum`: The output will be summed.
    """
    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 freq: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = "mean",
                 ) -> None:
        super(FocalLoss, self).__init__()
        assert alpha is None or freq is None, "Only one between alpha and freq can be passed"
        if alpha is None and freq is None:
            self.alpha = 1.0
        elif alpha is None:
            self.alpha = freq * freq.shape[0] / torch.sum(freq)
        elif freq is None:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self,
                yhat, y
                ) -> torch.Tensor:
        if self.alpha != 1.0:
            self.alpha = self.alpha.to(y.device)
        pt = F.softmax(yhat, dim=-1)
        logpt = F.log_softmax(yhat, dim=-1)
        if self.alpha == 1.0:
            alpha = torch.ones_like(y, device=y.device).unsqueeze(dim=-1)
        else:
            alpha = self.alpha[y].unsqueeze(dim=-1)
        floss = F.nll_loss(alpha * ((1 - pt) ** self.gamma) * logpt, y, reduction="none")
        if self.reduction == "none":
            return floss
        elif self.reduction == "mean":
            return torch.mean(floss)
        elif self.reduction == "sum":
            return torch.sum(floss)
