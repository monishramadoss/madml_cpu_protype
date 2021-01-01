from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from madml import tensor, zeros_like
from .module import Module
import numba as nb
import numpy as np

class ReLU(Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        super(ReLU, self).__init__()
        self.inplace = inplace
        self.out = None

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

    def forward_cpu(self, x: tensor) -> tensor:
        self.cache.append(x)
        tmp = np.maximum(x.host_data, 0)
        if self.inplace:
            self.cache.append(x)
            x.host_data = tmp
            return x
        else:
            y = zeros_like(x)
            self.cache.append(y)
            y.host_data = tmp
            return y

    def backward_cpu(self) -> tensor:
        x, y = self.cache
        dx, dy = x.gradient, y.gradient
        arr = dy.host_data.reshape(x.shape)
        arr[x.host_data <= 0] = 0
        x.gradient.host_data = arr.reshape(x.shape)
        if not self.inplace:
            y.zero_grad()
        return x
