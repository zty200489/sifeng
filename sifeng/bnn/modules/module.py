import torch
from torch.nn.modules.module import Module
from collections import OrderedDict
from typing import Any

__all__ = [
    "BayesianModule",
]

class BayesianModule(Module):
    """Base class for all bayesian neural network modules.

    Your bayesian models should also subclass this class. Be sure to override `forward()`.

    """
    def __init__(self):
        super(BayesianModule, self).__init__()
        super(BayesianModule, self).__setattr__('_bmodules', OrderedDict())
        self.register_buffer("log_prior", torch.tensor(0.0))
        self.register_buffer("log_posterior", torch.tensor(0.0))

    def __getattr__(self, name: str) -> Any:
        return super(BayesianModule, self).__getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, BayesianModule):
            bmodules = self.__dict__.get("_bmodules")
            bmodules[name] = value
        super(BayesianModule, self).__setattr__(name, value)

    def bmodules(self):
        return self.__dict__.get("_bmodules").values()

    def _log_prior(self):
        log_prior = self.log_prior
        for bmodule in self.bmodules():
            log_prior = log_prior + bmodule._log_prior()
        return log_prior

    def _log_posterior(self):
        log_posterior = self.log_posterior
        for bmodule in self.bmodules():
            log_posterior = log_posterior + bmodule._log_posterior()
        return log_posterior
