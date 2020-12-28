from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
from typing import Optional, List

from madml import tensor, zeros_like, zeros
from .module import Module
import numpy as np


def _size(shape: List[int]) -> int:
    size = 1
    for s in shape:
        size *= s
    return size


def softmax_util_cpu(x: tensor, y: tensor) -> tensor:
    eX = np.exp((x.host_data.T - np.max(x.host_data, axis=1)).T)
    y.host_data = (eX.T / eX.sum(axis=1)).T
    return y


def l1_reg(w: tensor, lam: float = 1e-3) -> float:
    return lam * np.sum(np.abs(w.host_data))


def l2_reg(w: tensor, lam: float = 1e-3) -> float:
    return .5 * lam * np.sum(w.host_data * w.host_data)


def dl1_reg(w: tensor, lam: float = 1e-3, esp: float = 1e-8) -> tensor:
    w.host_data *= lam
    w.host_data /= np.abs(w.host_data + esp)
    return w


def dl2_reg(w: tensor, lam: float = 1e-3) -> tensor:
    w.host_data *= lam
    return w


class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', backend=None) -> None:
        super(_Loss, self).__init__(backend)
        if size_average is not None or reduce is not None:
            self.reduction = None  # _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

    def regularize(self) -> float:
        reg_loss = 0.0
        params = self.parameters()
        for _, p in params.items():
            if self.reduction == 'mean' or self.reduction == 'l2':
                reg_loss += l2_reg(p.param)
            elif self.reduction == 'sum' or self.reduction == 'l1':
                reg_loss += l1_reg(p.param)
        return reg_loss


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.weight = weight


class CrossEntropyLoss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight=None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.loss = tensor([0], [1])
        self.batchsize = 1

    def forward_cpu(self, logit: tensor, target: tensor) -> tensor:
        self.batchsize = logit.shape[0]

        prob = zeros_like(logit)
        prob = softmax_util_cpu(logit, prob)
        t = np.zeros(logit.shape)
        tmp = target.host_data.astype(int)

        for i in range(self.batchsize):
            t_idx = tmp[i]
            t[i, t_idx] = 1

        reg_loss = self.regularize()
        self.loss.host_data = -np.sum(t * np.log(prob.host_data)) + reg_loss

        self.cache.append(logit)
        self.cache.append(target)
        self.cache.append(prob)
        return self.loss

    def backward_cpu(self) -> tensor:
        x, t, p = self.cache
        self.visited[self.loss.id] = True
        self.visited[t.id] = True
        dx = p.host_data
        for i in range(self.batchsize):
            t_idx = int(t.host_data[i][0])
            dx[i][t_idx] -= 1

        dx /= self.batchsize
        x.gradient.host_data = dx
        return x
