from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import madml
from madml import tensor
from .module import Module, get_parameters
from typing import Optional,

def regularization(reg_type='l2', lam=1e-3):
    reg_lambda = dict()[reg_type]
    
    params = get_parameters()
    reduction_list = []
    for p in params.keys():
        if params[p].is_weight():
            reduction_list.append(reg_lambda(params[p].data, lam))
    reg_loss = np.sum(reduction_list)

    return reg_loss

class _Loss(Module):
    reduction : str
    def __init__(self, size_average=None, reduce=None, reduction: str='mean', backend=None) -> None:
        super(_Loss, self).__init__(backend)
        if size_average is not None or reduce is not None:
            self.reduction = None #_Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

  
   
class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str='mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.weight = weight

class CrossEntropyLoss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index : int

    def __init__(self, weight: Optional[tensor]=None, size_average=None, ignore_index: int=-100,
                 reduce=None, reduction: str='mean') -> None:
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.exps = None
    def forward_cpu(self, logit: tensor, target: tensor) -> tensor:
        batchsize = logit.shape[0]
        if self.args is None:
            self.args = tensor()
        max = 0
        for i in range(logit.size):
            max = logit.host_data[i] if logit.host_data[i] > max else max

        for b in range(batchsize):
            prob = []
            for r in range(logit.size // batchsize):
                prob[r]
                
            

        exps = np.exp(logit - np.max(logit))
        prob = exps / np.sum(exps, axis=0)
        log_like = -np.log(prob[range(m), target])
        data_loss = np.sum(log_like) / m
        reg_loss = regularization(reg_type='l2', lam=1e-3)
        self.cache = [logit, target, prob, batchsize]
        return np.array([data_loss]) + reg_loss

    def backward_cpu(self) -> tensor:
        logit, target, grad_y, m = self.cache
        grad_y[range(m), target] -= 1.
        grad_y /= m
        return grad_y