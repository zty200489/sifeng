import torch
from .module import BayesianModule

__all__ = [
    "Sequential",
]

class Sequential(BayesianModule):
    """A sequential container

    The same as `torch.nn.Sequential`, except that bayesian modules inside will automatically calcu
    late their prior and posterior distributions.

    Parameters
    ----------
    *args:
        The sequence of `BayesianModules`

    """
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self.sequence = torch.nn.Sequential(*args)
        dict = self.__dict__.get("_bmodules")
        for idx, module in enumerate(args):
            if isinstance(module, BayesianModule):
                dict[str(idx)] = module

    def forward(self, x):
        return self.sequence(x)
