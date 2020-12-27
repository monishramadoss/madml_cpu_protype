from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
from typing import Optional, List

from madml import tensor, zeros_like, zeros
from .module import Module


def _size(shape: List[int]) -> int:
    size = 1
    for s in shape:
        size *= s
    return size


class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', backend=None) -> None:
        super(_Loss, self).__init__(backend)
        if size_average is not None or reduce is not None:
            self.reduction = None  # _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


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
        self.cache.append(logit)
        self.cache.append(target)
        self.cache.append(prob)
        return self.loss

    def backward_cpu(self) -> tensor:
        x, t, p = self.cache
        self.visited[self.loss.id] = True
        self.visited[t.id] = True
        dx = x.gradient
        assert(dx.size == x.size)
        return x
