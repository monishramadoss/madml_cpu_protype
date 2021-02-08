from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List, Optional

from madml import tensor

global parameter_cache

parameter_cache = []

DEBUG = False


class Parameter(object):
    param: tensor
    optimizer_stuff: Optional[List[tensor]]
    device: str
    shared_devices: bool

    def __init__(self, init_fn, shape: List[int], on_gpu: bool = False, shared_devices: bool = False) -> None:
        self.param = init_fn(shape)
        self.optimizer_stuff = []
        self.device = 'gpu' if on_gpu else 'cpu'
        self.shared_devices = shared_devices
        parameter_cache.append(self)

    def zero_grad(self, ) -> None:
        for i in range(self.param.size):
            self.param.grad_data[i] = 0

    def reshape(self, shape: List[int]) -> None:
        self.param.reshape(shape)


class Module(object):
    def __init__(self, backend=None):
        self.cache = []
        self.backend = backend
        self.registered = False
        self.visited = {}
        self.id = id(self)
        self.y = None
        self.print_out_flag = False

    def forward(self, *args, **kwargs) -> tensor:
        return self.forward_cpu(*args, **kwargs)

    def backward(self):
        x = self.backward_cpu()
        if DEBUG:
            self.print_l()
        if isinstance(x, tensor):
            x.reset_shape()
        return x

    def forward_cpu(self, *args, **kwargs) -> tensor:
        pass

    def backward_cpu(self) -> tensor:
        pass

    def __call__(self, *args, **kwargs):
        # print(type(self), 'forward')
        y = self.forward(*args, **kwargs)
        if isinstance(y, tuple) or isinstance(y, list):
            for x in y:
                self.visited[x.id] = False
                if self not in x.parent:
                    x.parent += [self]
                x.zero_grad()
        else:
            self.visited[y.id] = False
            if self not in y.parent:
                y.parent += [self]
            y.zero_grad()
        for x in args:
            self.visited[x.id] = False
            if self not in x.children:
                x.children += [self]
            x.zero_grad()

        if not self.registered:
            self.registered = True
        # if isinstance(y, tensor):
        #      print('\t', y.shape, y.host_data.max(), y.host_data.min())
        return y

    @staticmethod
    def parameters() -> List[Parameter]:
        return parameter_cache

    def print_l(self):
        print(type(self), end=': ')
        for t in self.cache:
            if isinstance(t, tensor):
                print(t.shape, end=' ')
        print()
        pass
