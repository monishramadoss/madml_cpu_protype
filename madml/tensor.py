from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import struct
from typing import List, Union
import numpy as np

# from .nn.module import module_cache, execution_order
_gradients = {}
_tensors = []


def _convert_to_float(size: int, arr: List[bytes]) -> List[float]:
    ret_data = []
    ret_data.extend([bytearray(arr[i:i + 4]) for i in range(0, size, 4)])
    for i in range(len(ret_data)):
        ret_data[i] = struct.unpack("f", ret_data[i])
    return ret_data


class tensor(object):
    shape: List[int]
    init_shape: List[int]
    host_memory: np.ndarray
    device_memory: List[Union[float, int, bytes, bool]]
    on_device: bool
    id: int

    def __init__(self, data: Union[List[Union[float, int, bytes, bool]], np.ndarray], shape: List[int] = []) -> None:
        if isinstance(data, np.ndarray):
            self.host_memory = data
            self.shape = list(data.shape)
            self.device_memory = [data.ravel()[i] for i in range(data.size)]
        else:
            self.host_memory = np.array(data).reshape(shape)
            self.shape = shape
            self.device_memory = [data[i] for i in range(len(data))]

        self.init_shape = self.shape
        self.size = 1
        for s in self.shape:
            self.size *= s

        self.on_device = False
        self.parent = []
        self.children = []
        self.id = id(self)
        assert (len(self.shape) > 0)
        assert (self.host_memory.size == self.size)

    def __len__(self):
        return self.shape[0]

    def T(self):
        assert len(self.shape) == 2
        self.shape = [self.shape[1], self.shape[0]]
        return self

    def numpy(self):
        return self.host_memory

    def __getitem__(self, idx: int):
        assert (self.shape[0] > idx)
        new_data = self.host_memory[idx]
        new_shape = self.shape[:idx]
        return tensor(new_data, new_shape)

    def __setitem__(self, key: int, value) -> None:
        assert (self.size > key)
        assert (type(value) == type(self))
        self.host_memory[key] = value.host_memory

    def copy(self):
        return tensor(self.host_memory, self.shape)

    def reshape(self, shape: List[int]):
        self.host_memory.reshape(shape)
        self.shape = list(self.host_memory.shape)

    @property
    def gradient(self):
        if self.id not in _gradients.keys():
            _gradients[self.id] = tensor([0 for _ in range(self.size)], self.init_shape)
        assert (self.size == _gradients[self.id].size)
        return _gradients[self.id]

    @property
    def grad_data(self):
        grad = self.gradient.host_data
        return grad.ravel()

    @property
    def host_data(self):
        return self.host_memory

    @property
    def device_data(self):
        return self.device_memory

    def backward(self):
        for x in reversed(self.parent):
            if not x.visited[self.id]:
                y = x.backward()
                x.visited[self.id] = True
                if isinstance(y, tensor):
                    y.backward()
        self.parent.clear()
        self.children.clear()
        return

    def numpy(self):
        return self.host_memory

    def reset(self):
        self.shape = self.init_shape
        if self.id in _gradients.keys():
            self.gradient.reshape(self.init_shape)

    def flatten(self):
        s = 1
        for S in self.shape[1:]:
            s *= S
        self.reshape([self.shape[0], s])
        assert (self.shape[0] * s == self.size)
        if self.id in _gradients.keys():
            self.gradient.reshape(self.init_shape)
