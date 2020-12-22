from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List

from madml import tensor, zeros

global execution_order
global parameter_cache
global module_cache

module_cache = {}
parameter_cache = {}
execution_order = []


class Parameter(object):
    param: tensor
    device: str
    shared_devices: bool

    def __init__(self, init_fn, shape: List[int], on_gpu: bool = False, shared_devices: bool = False) -> None:
        self.param = init_fn(shape)
        self.device = 'gpu' if on_gpu else 'cpu'
        self.shared_devices = shared_devices
        self.velocity = zeros(shape)
        parameter_cache[id(self)] = self

    def zero_grad(self, ) -> None:
        for i in range(self.param.size):
            self.param.grad_data.host_data[i] = 0

    def reshape(self, shape: List[int]) -> None:
        self.param.reshape(shape)


class Module(object):
    def __init__(self, backend=None):
        self.cache = []
        self.backend = backend
        self.registered = False
        self.visited = {}
        module_cache[id(self)] = self

    def forward(self, *args, **kwargs):
        return self.forward_cpu(*args, **kwargs)

    def backward(self):
        return self.backward_cpu()

    def forward_cpu(self, *args, **kwargs):
        raise NotImplementedError

    def backward_cpu(self):
        pass

    def __call__(self, *args, **kwargs):
        print(type(self), end=' ')
        y = self.forward(*args, **kwargs)

        if not self.registered:
            if isinstance(y, tuple) or isinstance(y, list):
                for x in y:
                    x.parent += [self]
                    self.visited[id(x)] = False
            else:
                y.parent += [self]
                self.visited[id(y)] = False
            for x in args:
                x.children += [self]
                self.visited[id(x)] = False
            execution_order.append(self)
            self.registered = True

        if isinstance(y, tensor):
            print(y.shape)
        return y

    def parameters(self):
        return parameter_cache
