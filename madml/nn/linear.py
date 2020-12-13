from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .module import Module
from madml import tensor, normal, zeros


def matmul(x: tensor, w: tensor, y: tensor, use_grad: bool = False):
    for m in range(x.shape[0]):
        for n in range(w.shape[1]):
            acc = 0
            for k in range(w.shape[0]):
                acc += x.host_data[m * w.shape[0] + k] * w.host_data[k * w.shape[1] + n]
            if use_grad:
                y.grad_data[m * w.shape[0] + n] = acc
            else:
                y.host_data[m * w.shape[0] + n] = acc


class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = normal()([out_features, in_features], True)
        self.bias = zeros([out_features], True) if bias else None

    def forward_cpu(self, x: tensor) -> tensor:
        y = zeros([x.shape[0], self.out_features])
        matmul(x, self.weight, y)

        if self.bias is not None:
            for m in range(y.shape[0]):
                for n in range(y.shape[1]):
                    y[m * self.out_features + n] += self.bias.host_data[n]

        self.cache.append(x)
        return y

    def backward_cpu(self, dy: tensor) -> tensor:
        x = self.cache[0]
        dx = zeros(x.shape)
        dx.link(x)
        matmul(dy, self.weight, dx)
        matmul(dy.T(), x, self.weight, True)
        return dx
