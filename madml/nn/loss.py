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


def regularization(reg_type='l2', lam=1e-3):
    return 1


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
        self.args = None
        self.ignore_index = ignore_index
        self.exps = None
        self.loss = tensor([0], [1])
        self.batchsize = 1

    def forward_cpu(self, logit: tensor, target: tensor) -> tensor:
        self.batchsize = logit.shape[0]
        if self.args is None:
            pass

        # MAX
        max = 0
        for i in range(logit.size):
            max = logit.host_data[i] if logit.host_data[i] > max else max

        # REDUCE_SUM
        reduce_sum = zeros(logit.shape[:1])
        upper = _size(logit.shape[:1])
        lower = _size(logit.shape[1:])
        for i in range(upper):
            acc = 0
            for j in range(lower):
                acc += logit.host_data[i * lower + j]
            reduce_sum.host_data[i] = acc

        # PROBABILITY
        self.prob = zeros_like(logit)
        for i in range(upper):
            for j in range(lower):
                self.prob.host_data[i * lower + j] = math.exp(logit.host_data[i * lower + j] - max)
            mu = 0
            for j in range(lower):
                mu += self.prob.host_data[i * lower + j]
            for j in range(lower):
                self.prob.host_data[i * lower + j] /= mu

        # LOG FN
        log_like = zeros_like(logit)
        for i in range(upper):
            for j in range(lower):
                log_like.host_data[i * lower + j] = -math.log(self.prob.host_data[i * lower + target.host_data[i]])

        self.loss.host_data[0] = 0
        for x in range(logit.size):
            self.loss.host_data[0] += logit.host_data[x] / self.batchsize

        self.cache.append(logit)
        self.cache.append(target)
        return self.loss

    def backward_cpu(self) -> tensor:
        logit, target = self.cache
        print(self.loss.parent, self.loss.children)
        self.visited[id(self.loss)] = True
        self.visited[id(target)] = True
        upper = _size(logit.shape[:1])
        lower = _size(logit.shape[1:])

        for i in range(upper):
            self.prob.host_data[i * lower + target.host_data[i]] -= 1.
            self.prob.host_data[i * lower + target.host_data[i]] /= self.batchsize

        return logit
