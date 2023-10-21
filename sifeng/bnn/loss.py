import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, Categorical
from . import BayesianModule

__all__ = [
    "BayesianLoss",
    "ContinuousELBOLoss",
    "DiscreteELBOLoss",
]

class BayesianLoss(nn.Module):
    """The Bayesian Loss Function

    The Bayesian loss compares the prior and posterior distribution of the parameters in the model
    to help bayesian models diffuse when faced with OOS data.

    Parameters
    ----------
    model: BayesianModule
        Youe model, be sure to pass an instance rather than the class

    """
    def __init__(self,
                 model: BayesianModule,
                 ) -> None:
        super(BayesianLoss, self).__init__()
        self.model = model

    def forward(self) -> torch.Tensor:
        return self.model._log_posterior() - self.model._log_prior()

class ContinuousELBOLoss(nn.Module):
    """The ELBO loss for regression (Continuous) tasks

    Parameters
    ----------
    noise: float, default `0.01`
        The tolerable noise level, or namely, the std in the gaussian mixture model for approximati
    ng the distribution function.

    """
    def __init__(self,
                 noise: float = 0.01,
                 ) -> None:
        super(ContinuousELBOLoss, self).__init__()
        self.noise = noise

    def forward(self, yhat, y) -> torch.Tensor:
        logp = Normal(yhat, self.noise).log_prob(y)
        return -torch.sum(torch.logsumexp(logp, dim=0) - np.log(logp.shape[0]))

class DiscreteELBOLoss(nn.Module):
    """The ELBO loss for classification (Discrete) tasks

    """
    def __init__(self,
                 ) -> None:
        super(DiscreteELBOLoss, self).__init__()

    def forward(self, yhat, y) -> torch.Tensor:
        logp = Categorical(logits=yhat).log_prob(y.squeeze(-1))
        return -torch.sum(torch.logsumexp(logp, dim=0) - np.log(logp.shape[0]))
