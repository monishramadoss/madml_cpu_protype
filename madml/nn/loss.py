from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List

import numpy as np

from madml import tensor
from .module import Module


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
        self.loss = tensor([0], [1])

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

    def __init__(self, weight=None, size_average=None, ignore_index: int = None,
                 reduce=None, reduction: str = 'mean') -> None:
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

        self.batchsize = 1

    def forward_cpu(self, logit: tensor, target: tensor) -> tensor:
        assert (len(logit.shape) != 1)

        N = logit.shape[0]
        C = logit.shape[1]
        x = logit.host_data
        t = target.host_data

        max_x = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - max_x)
        p = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        inp = np.log(p)

        gather_weight = None
        if self.weight is not None:
            gather_weight = np.take(self.weight, t, mode='clip')
            if self.ignore_index is not None:
                gather_weight = np.where(t == self.ignore_index, 0, gather_weight).astype(dtype=np.float32)
        elif self.ignore_index is not None:
            gather_weight = np.where(t == self.ignore_index, 0, 1).astype(dtype=np.float32)

        if len(inp.shape) != 3:
            inp = inp.reshape([N, C, -1])
            t = t.reshape([N, -1])
        D = inp.shape[2]

        neg_gather_element_input = np.zeros((N, D), dtype=np.float32)
        for i in range(N):
            for d in range(D):
                if t[i][d] != self.ignore_index:
                    idx = int(t[i][d])
                    neg_gather_element_input[i][d] = -inp[i][idx][d]

        loss = neg_gather_element_input
        if self.reduction == 'mean':
            loss = np.mean(loss)
        elif self.reduction == 'sum':
            loss = np.sum(loss)

        self.loss.host_data = loss + self.regularize()

        self.cache.append(logit)
        self.cache.append(target)
        self.cache.append(p)
        return self.loss

    def backward_cpu(self) -> tensor:
        x, t, p = self.cache
        self.visited[self.loss.id] = True
        self.visited[t.id] = True
        t = t.host_data.reshape([x.shape[0], -1])
        dx = p
        for i in range(self.batchsize):
            for d in range(t.shape[1]):
                t_idx = int(t[i][d])
                dx[i][t_idx] -= 1

        dx /= self.batchsize
        x.gradient.host_data = dx
        return x


class MSELoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> tensor:
        super(MSELoss, self).__init__(size_average, reduce, reduction)

    def forward_cpu(self, logit: tensor, target: tensor) -> tensor:
        m = logit.shape[0]
        data_loss = (np.square(logit.host_data - target.host_data)).mean(axis=0)
        data_loss /= m

        if self.reduction == 'mean':
            loss = np.mean(data_loss)
        elif self.reduction == 'sum':
            loss = np.sum(data_loss)

        self.loss.host_data = loss + self.regularize()

        self.cache.append(logit)
        self.cache.append(target)
        self.cache.append(m)
        return self.loss

    def backward_cpu(self) -> tensor:
        x, t, m = self.cache
        grad_y = 2 * (x.host_data - t.host_data)
        grad_y /= m
        x.gradient.host_data = grad_y
        return x
