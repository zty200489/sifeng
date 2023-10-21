import numpy as np
import torch
from torch.nn import Parameter, init
from torch.nn.functional import softplus
from torch.distributions import Normal
from .module import BayesianModule

__all__ = [
    "Linear",
]

class Linear(BayesianModule):
    """The bayesian linear transformation.

    Applies a linear transformation to the incoming data :math:`y = x\mathbf{A}^T + \mathbf{b}`,
    where :math:`\mathbf{A}` and :math:`\mathbf{b}` are random variables.

    Parameters
    ----------
    in_features: int
        The size of each input sample.
    out_features: int
        The size of each output sample.
    bias: bool, default `True`
        If set to `False`, the layer will not learn an additive bias.
    eps: float, default `1e-5`
        The minimal epsilon added when calculating the standard deviations.

    Input shape
    -----------
    [sample * ?? * in_features]

    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 prior_var: float = 10.0,
                 eps: float = 1e-5,
                 ) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.prior_var = prior_var
        self.eps = eps

        self.weight_mean = Parameter(torch.empty(in_features, out_features))
        self.weight_rho = Parameter(torch.zeros(in_features, out_features))
        if bias:
            self.bias_mean = Parameter(torch.empty(out_features))
            self.bias_rho = Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self,
                         ) -> None:
        init.kaiming_uniform_(self.weight_mean, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mean)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias_mean, -bound, bound)

    def forward(self,
                x: torch.Tensor, # [sample * ?? * in_features]
                ) -> torch.Tensor:
        sample, shape = x.shape[0], x.shape[1:-1]
        x = x.flatten(1, -2) # [sample * full_shape * in_features]
        weight_sigma = softplus(self.weight_rho) + self.eps # [out_features * in_features]
        weight = torch.randn([sample, self.in_features, self.out_features], device=x.device) * weight_sigma + self.weight_mean # [sample * out_features * in_features]
        if self.bias:
            bias_sigma = softplus(self.bias_rho) + self.eps # [out_features]
            bias = torch.randn([sample, self.out_features], device=x.device) * bias_sigma + self.bias_mean # [sample * out_features]

        if self.bias:
            if self.training:
                self.log_prior = torch.sum(torch.logsumexp(Normal(0, self.prior_var).log_prob(weight), dim=0) - np.log(weight.shape[0])) + \
                                 torch.sum(torch.logsumexp(Normal(0, self.prior_var).log_prob(bias), dim=0) - np.log(bias.shape[0]))
                self.log_posterior = torch.sum(torch.logsumexp(Normal(self.weight_mean, weight_sigma).log_prob(weight), dim=0) - np.log(weight.shape[0])) + \
                                     torch.sum(torch.logsumexp(Normal(self.bias_mean, bias_sigma).log_prob(bias), dim=0) - np.log(bias.shape[0]))
            x = torch.bmm(x, weight) + bias.unsqueeze(1) # [sample * full_shape * out_features]
            return x.unflatten(1, shape)
        else:
            if self.training:
                self.log_prior = torch.sum(torch.logsumexp(Normal(0, self.prior_var).log_prob(weight), dim=0) - np.log(weight.shape[0]))
                self.log_posterior = torch.sum(torch.logsumexp(Normal(self.weight_mean, weight_sigma).log_prob(weight), dim=0) - np.log(bias.shape[0]))
            x = torch.bmm(x, weight) # [sample * full_shape * out_features]
            return x.unflatten(1, shape)
