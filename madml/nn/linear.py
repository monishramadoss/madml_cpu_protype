from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .module import Module, Parameter
from madml import tensor, normal, zeros


def matmul(x: tensor, w: tensor, y: tensor, use_grad: bool = False):
    assert (x.shape[1] == w.shape[0])

    if len(x.shape) == 2:
        for b in range(x.shape[0]):
            for m in range(x.shape[1]):
                for n in range(w.shape[1]):
                    if use_grad:
                        y.grad_data[b * w.shape[1] + n] += x.host_data[b * x.shape[1] + m] * w.host_data[
                            m * w.shape[1] + n]
                    else:
                        y.host_data[b * w.shape[1] + n] += x.host_data[b * x.shape[1] + m] * w.host_data[
                            m * w.shape[1] + n]

    elif len(x.shape) == 3:
        for b in range(x.shape[0]):
            for m in range(x.shape[1]):
                for n in range(w.shape[1]):
                    for k in range(w.shape[0]):
                        if use_grad:
                            y.grad_data[b * w.shape[1] + n] += x.host_data[b * x.shape[1] + m] * w.host_data[
                                m * w.shape[1] + n]
                        else:
                            y.host_data[b * w.shape[1] + n] += x.host_data[b * x.shape[1] + m] * w.host_data[
                                m * w.shape[1] + n]


class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(normal(), [in_features, out_features])
        self.bias = Parameter(zeros, [out_features]) if bias else None

    def forward_cpu(self, x: tensor) -> tensor:
        y = zeros([x.shape[0], self.out_features])
        matmul(x, self.weight.param, y)

        if self.bias is not None:
            for m in range(y.shape[0]):
                for n in range(y.shape[1]):
                    y[m * self.out_features + n] += self.bias.param.host_data[n]

        self.cache.append(x)
        return y

    def backward_cpu(self, dy: tensor) -> tensor:
        x = self.cache[0]
        dx = zeros(x.shape)
        dx.link(x)
        matmul(dy, self.weight.param, dx)
        matmul(dy.T(), x, self.weight.param, True)
        return dx
