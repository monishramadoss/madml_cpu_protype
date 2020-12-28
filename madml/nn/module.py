from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List, Dict

from madml import tensor, zeros

global execution_order
global parameter_cache
global module_cache

module_cache = {}
parameter_cache = {}
execution_order = []
DEBUG = False


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
        module_cache[self.id] = self

    def forward(self, *args, **kwargs):
        return self.forward_cpu(*args, **kwargs)

    def backward(self):
        # print(type(self), 'backward')
        # for x in self.cache:
        #     if isinstance(x, tensor):
        #         print(x.gradient.host_data.max(), end=' ')
        dx = self.backward_cpu()
        # if isinstance(dx, tensor):
        #     print('\t', dx.shape, dx.host_data.max())
        return dx

    def forward_cpu(self, *args, **kwargs):
        raise NotImplementedError

    def backward_cpu(self):
        pass

    def __call__(self, *args, **kwargs):
        # print(type(self), 'forward')
        self.cache.clear()
        y = self.forward(*args, **kwargs)

        if isinstance(y, tuple) or isinstance(y, list):
            for x in y:
                self.visited[x.id] = False
                if self not in x.parent:
                    x.parent += [self]
        else:
            self.visited[y.id] = False
            if self not in y.parent:
                y.parent += [self]
        for x in args:
            self.visited[x.id] = False
            if self not in x.children:
                x.children += [self]

        if not self.registered:
            execution_order.append(self)
            self.registered = True

        # if isinstance(y, tensor):
        #      print('\t', y.shape, y.host_data.max())
        return y

    def parameters(self) -> Dict[int, Parameter]:
        x = self.id
        return parameter_cache
