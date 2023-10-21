import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import logging
from typing import Optional, Callable, Union

g_device = "cuda:0" if torch.cuda.is_available() else "cpu"

__all__ = [
    "VerboseBase",
    "dl_trainer",
    "dl_evaluator",
]

class VerboseBase:
    """The base for all Verbosers

    Functions
    ---------
    epoch_end: [int, str] -> Optional[float]:
        Will be called when an epoch ends, now you can summarize your results and output/log your
    evaluation, additionally, you may also return your metric to override early stopping (Note that
    the metric also assumes less being better, so metrics like accuracy should be negatived before
    returning)
    eval_iter: [torch.Tensor, torch.Tensor] -> None:
        Will be called each epoch, and you may calculate whatever metric you want

    """
    def __init__(self):
        pass

    def epoch_end(self,
                  epoch: int,
                  mode: str,
                  ) -> Optional[float]:
        raise NotImplementedError

    def eval_iter(self,
                  yhat: torch.Tensor,
                  y: torch.Tensor
                  ) -> None:
        raise NotImplementedError

class dl_trainer:
    """The torch model trainer

    Parameters
    ----------
    model: torch.nn.Module
        The model to be trained
    loss: Callable[[torch.nn.tensor, torch.nn.tensor], torch.tensor]
        The loss function
    optimizer: torch.optim.Optimizer
        The BP optimizer
    epoch: int
        Number of epochs to train
    lr: float,
        The learning rate
    lr_scheduler: Optional[Callable[[int], float]], default `None`
        A function that computes a multiplicative factor given an integer parameter epoch, `None`
    to disable
    batch_size: Union[int, Callable[[int], int]], default `64`
        A function that computes batch size given an integer parameter epoch, or a constant batch
    size, only enable when passing a
    `torch.utils.data.Dataset` to the trainer
    mini_batch: Union[int, Callable[[int], int]], default `1`
        A function that computes mini batch size given an integer parameter epoch, or a constant
    mini batch size, useful when you have a small GPU
    memory size but want much larger batch sizes
    patience: int, default `-1`:
        The patience parameter for early stopping, set to `-1` to disable
    delta: float, default `0`:
        The minimal loss improvement required for the model to be considered improving
    logger: Optional[logging.Logger], default `None`
        A logger for debug-level information
    verbose: Optional[VerboseBase], default `None`
        The verbose evaluator for epoch level data
    eval_train: bool, default `False`
        Whether to eval the train set every epoch
    device: torch.device, default `"cuda:0" if torch.cuda.is_available() else "cpu"`
        On which device to train the model
    *args, **kwargs:
        Additional args for optimizer
    """
    def __init__(self,
                 model: nn.Module,
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 optimizer: torch.optim.Optimizer,
                 epoch: int,
                 lr: float,
                 lr_scheduler: Optional[Callable[[int], float]] = None,
                 batch_size: Union[int, Callable[[int], int]] = 64,
                 mini_batch: Union[int, Callable[[int], int]] = 1,
                 patience: int = -1,
                 delta: float = 0,
                 logger: Optional[logging.Logger] = None,
                 verbose: Optional[VerboseBase] = None,
                 eval_train: bool = False,
                 device: torch.device = "cuda:0" if torch.cuda.is_available() else "cpu",
                 *args, **kwargs,
                 ) -> None:
        self.model = model
        self.loss = loss
        self.lr = lr
        self.optimizer = optimizer(filter(lambda layer: layer.requires_grad, self.model.parameters()), lr=lr, *args, **kwargs)
        self.epoch = epoch
        self.lr_lambda = lr_scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_scheduler) if lr_scheduler is not None else None
        self.batch_size = batch_size
        self.mini_batch = mini_batch
        self.patience = patience
        self.delta = delta
        self.logger = logger
        self.verbose = verbose
        self.eval_train = eval_train
        self.device = device

    def train(self,
              train_data: Union[Dataset, DataLoader],
              valid_data: Union[Dataset, DataLoader],
              *args, **kwargs) -> None:
        """Train the model

        Parameters
        ----------
        train_data: Union[Dataset, DataLoader]
            The training data
        valid_date: Union[Dataset, DataLoader]
            The validation data
        *args, **kwargs:
            Additional parameters for `torch.utils.data.DataLoader` if passing a `torch.utils.data.Dataset`
        """
        self.logger.info("Begin training model. Training params: loss = {}; optimizer = {}.".format(self.loss, self.optimizer)) if self.logger else None # INFO-LOG
        torch.cuda.empty_cache()
        self.model.to(self.device)
        best_loss, best_metric, patience = np.inf, np.inf, self.patience
        self.model.to(self.device)
        for EPOCH in tqdm(range(self.epoch), desc="training", unit="epoch"):
            self.logger.info("Start training@EPOCH {}. Current params: lr = {}{}{}; patience = {}/{}".format(EPOCH, self.lr, " * {}".format(self.lr_lambda(EPOCH)) if self.lr_lambda else "", "; batch size = {} * {}".format(self.batch_size(EPOCH) if callable(self.batch_size) else self.batch_size, self.mini_batch(EPOCH) if callable(self.mini_batch) else self.mini_batch) if isinstance(train_data, Dataset) else "", patience, self.patience)) if self.logger else None # INFO-LOG
            if patience > 0:
                patience = patience - 1
            self.train_epoch(EPOCH, train_data, *args, **kwargs)
            if self.eval_train:
                loss = self.eval_epoch(EPOCH, train_data, *args, **kwargs)
                self.logger.info(f"Finished train-eval@EPOCH {EPOCH}, current loss = {loss:.8f}") if self.logger else None # INFO-LOG
                if self.verbose is not None:
                    metric = self.verbose.epoch_end(EPOCH, "train")
            loss = self.eval_epoch(EPOCH, valid_data, *args, **kwargs)
            if loss < best_loss - self.delta:
                self.logger.info(f"Finished validing@EPOCH {EPOCH}, current loss = {loss:.8f}, loss update {best_loss:.8f} -> {loss:.8f}") if self.logger else None # INFO-LOG
                best_loss = loss
                patience = self.patience
            else:
                self.logger.info(f"Finished validing@EPOCH {EPOCH}, current loss = {loss:.8f}") if self.logger else None # INFO-LOG
            if self.verbose is not None:
                metric = self.verbose.epoch_end(EPOCH, "valid")
                if metric is not None and metric < best_metric:
                    self.logger.info(f"Verbose metric: {metric:.8f}, metric update {best_metric:.8f} -> {metric:.8f}.") if self.logger else None # INFO LOG
                    best_metric = metric
                    patience = self.patience
                else:
                    self.logger.info(f"Verbose metric: {metric:.8f}.") if self.logger else None # INFO LOG
            if patience == 0:
                break

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def train_epoch(self,
                    epoch: int,
                    train_data: Union[Dataset, DataLoader],
                    *args, **kwargs
                    ) -> None:
        if isinstance(train_data, DataLoader):
            train_loader = train_data
        elif isinstance(self.batch_size, int):
            train_loader = DataLoader(train_data, batch_size=self.batch_size, *args, **kwargs)
        elif callable(self.batch_size):
            train_loader = DataLoader(train_data, batch_size=self.batch_size(epoch), *args, **kwargs)
        else:
            raise ValueError
        mini_batch, countr = self.mini_batch if isinstance(self.mini_batch, int) else self.mini_batch(epoch), 0
        self.model.train()
        for idx, packet in enumerate(train_loader):
            x, y = packet[0].to(self.device), packet[1].to(self.device)
            yhat = self.model(x)
            loss = self.loss(yhat, y) / mini_batch
            if countr == 0:
                self.optimizer.zero_grad()
            loss.backward()
            countr = countr + 1
            if countr == mini_batch:
                self.optimizer.step()
                self.logger.debug("Training iter {}, loss = {:.8f}, optimizer stepped.".format(idx, loss.data * mini_batch)) if self.logger else None # DEBUG-LOG
                countr = 0
            else:
                self.logger.debug("Training iter {}, loss = {:.8f}.".format(idx, loss.data * mini_batch)) if self.logger else None # DEBUG-LOG

    def eval_epoch(self,
                   epoch: int,
                   valid_data: Union[Dataset, DataLoader],
                   *args, **kwargs
                   ) -> float:
        if isinstance(valid_data, DataLoader):
            valid_loader = valid_data
        elif isinstance(self.batch_size, int):
            valid_loader = DataLoader(valid_data, batch_size=self.batch_size, *args, **kwargs)
        elif callable(self.batch_size):
            valid_loader = DataLoader(valid_data, batch_size=self.batch_size(epoch), *args, **kwargs)
        else:
            raise ValueError
        accum_loss, accum_batch = 0, 0
        self.model.eval()
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                loss = self.loss(yhat, y)
                accum_loss = accum_loss + loss.data * y.shape[0]
                accum_batch = accum_batch + y.shape[0]
                if self.verbose is not None:
                    self.verbose.eval_iter(yhat=yhat, y=y)
        return accum_loss / accum_batch

class dl_evaluator:
    """The torch model evaluator

    Parameters
    ----------
    model: torch.nn.Module
        The trained model.
    verbose: Optional[VerboseBase], default `None`
        The verbose evaluator for infomation output
    device: torch.device, default `"cuda:0" if torch.cuda.is_available() else "cpu"`
        On which device to evaluate the model
    """
    def __init__(self,
                 model: nn.Module,
                 verbose: Optional[VerboseBase] = None,
                 device: torch.device = "cuda:0" if torch.cuda.is_available() else "cpu",
                 ) -> None:
        self.model = model
        self.verbose = verbose
        self.device = device

    def eval(self,
             valid_data: Union[Dataset, DataLoader],
             *args, **kwargs,
             ) -> float:
        if isinstance(valid_data, DataLoader):
            valid_loader = valid_data
        else:
            valid_loader = DataLoader(valid_data, *args, **kwargs)
        self.model.eval()
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                if self.verbose is not None:
                    self.verbose.eval_iter(yhat=yhat, y=y)
        self.verbose.epoch_end(0, "eval")
