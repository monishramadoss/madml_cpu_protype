from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from madml import tensor
from .module import Module


class ReLU(Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = True) -> None:
        super(ReLU, self).__init__()
        self.inplace = inplace
        self.out = None

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

    def forward_cpu(self, x: tensor) -> tensor:
        self.cache.append(x)
        if self.inplace:
            for i in range(x.size):
                if x.host_data[i] <= 0:
                    x.host_data[i] = 0
            return x
        else:
            if self.out is None:
                self.out = tensor([0. for _ in range(x.size)], x.shape)
            for i in range(x.size):
                self.out.host_data[i] = 0 if x.host_data[i] <= 0 else x.host_data[i]
            return self.out

    def backward_cpu(self) -> tensor:
        x = self.cache[0]
        if self.inplace:
            for i in range(x.size):
                if x.host_data[i] <= 0:
                    x.gradient.host_data[i] = 0
                else:
                    x.gradient.host_data[i] = x.host_data[i]
            return x
        else:
            return x
