from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import numpy as np

from madml import tensor, zeros, kaiming_uniform
from .module import Module, Parameter


class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(kaiming_uniform(a=math.sqrt(5), nonlinearity='linear'), [in_features, out_features])
        self.bias = Parameter(zeros, [out_features]) if bias else None

    def forward_cpu(self, x: tensor) -> tensor:
        y = zeros([x.shape[0], self.out_features])

        y.host_data = x.host_data @ self.weight.param.host_data
        if self.bias is not None:
            y.host_data += self.bias.param.host_data

        self.cache.append(x)
        self.cache.append(y)
        return y

    def backward_cpu(self) -> tensor:
        x, y = self.cache
        dx, dy = x.gradient, y.gradient

        if self.bias is not None:
            self.bias.param.gradient.host_data = np.sum(dy.host_data, axis=0)

        self.weight.param.gradient.host_data = x.host_data.T @ dy.host_data
        x.gradient.host_data = dy.host_data @ self.weight.param.host_data.T
        self.print()
        y.zero_grad()
        return x

    def print(self) -> None:
        x, y = self.cache
        print('linear:', x.shape, y.shape)
        print(' max input:', x.host_data.max(), 'g', x.gradient.host_data.max(),
              ' weight:', self.weight.param.host_data.max(), 'g', self.weight.param.gradient.host_data.max(),
              ' output:', y.host_data.max(), 'g', y.gradient.host_data.max())
        print(' min input:', x.host_data.min(), 'g', x.gradient.host_data.min(),
              ' weight:', self.weight.param.host_data.min(), 'g', self.weight.param.gradient.host_data.min(),
              ' output:', y.host_data.max(), 'g', y.gradient.host_data.min())
