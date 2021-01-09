from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from madml import tensor, zeros_like
from .module import Module


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
        tmp = x.host_data > 0
        data = x.host_data * tmp
        self.cache.append(tmp)
        if self.inplace:
            self.cache.append(x)
            x.host_data = data
            return x
        else:
            y = zeros_like(x)
            self.cache.append(y)
            y.host_data = data
            return y

    def backward_cpu(self) -> tensor:
        x, tmp, y = self.cache
        dx, dy = x.gradient, y.gradient
        arr = dy.host_data.reshape(x.shape) * tmp
        x.gradient.host_data = arr.reshape(x.shape)
        self.print()
        if not self.inplace:
            y.zero_grad()
        return x


    def print(self) -> None:
        x, t, y = self.cache
        print('relu:', x.shape, y.shape)
        print(' max input:', x.host_data.max(), 'g', x.gradient.host_data.max(),
              ' output:', y.host_data.max(), 'g', y.gradient.host_data.max())
        print(' min input:', x.host_data.min(), 'g', x.gradient.host_data.min(),
              ' output:', y.host_data.max(), 'g', y.gradient.host_data.min())
