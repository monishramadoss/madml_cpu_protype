from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List
from madml import tensor, zeros


class Parameter(object):
    param: tensor
    device: str
    shared_devices: bool

    def __init__(self, init_fn, shape: List[int], on_gpu: bool = False, shared_devices: bool = False) -> None:
        self.param = init_fn(shape)
        self.device = 'gpu' if on_gpu else 'cpu'
        self.shared_devices = shared_devices
        self.velocity = zeros(shape)


class Module(object):
    def __init__(self):
        self.cache = []

    def forward(self, *args, **kwargs):
        print(type(self))
        return self.forward_cpu(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        raise NotImplementedError

    def backward_cpu(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
